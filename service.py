import os
import shutil
import pathlib
from collections import defaultdict

import requests
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from fastapi.templating import Jinja2Templates
from typing import Optional, List, Dict, Any
import uuid
from uuid import UUID
from urllib.parse import urlparse, unquote
import ffmpeg
from PIL import Image
import statistics
import pandas as pd
import numpy as np

from src.database import VectorDB, VectorDBLite
from src.hashers import *


db = VectorDBLite("./vector.db", "dim640", 640, recreate_collection=False)
# db.delete_collection()

UPLOAD_DIR = "downloaded_videos"
FRAME_DIR = "frames"
EXTRACTION_FPS_RATE = 8 # число кадров в секунду, которые мы извлекаем
hasher = MixHash()

app = FastAPI(
    title="Video Duplicate Checker API",
    version="1.0.0",
    description="API для проверки дубликатов видео"
)

class VideoLinkRequest(BaseModel):
    link: HttpUrl


class VideoLinkResponse(BaseModel):
    is_duplicate: bool
    duplicate_for: Optional[str] = None


def extract_filename(url):
    decoded_url = unquote(url)
    parsed_url = urlparse(decoded_url)
    path = parsed_url.path
    filename = os.path.basename(path).split('.')[0]
    return filename


def download_video(url: str) -> pathlib.Path:
    video_id = extract_filename(url)
    video_dir = os.path.join(UPLOAD_DIR, video_id)
    os.makedirs(video_dir, exist_ok=True)
    video_path = pathlib.Path(video_dir) / f"{video_id}.mp4"

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with pathlib.Path.open(video_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    return video_path



def remove_consecutive_duplicates(lst):
    result = list(set(lst))
    return result


class FrameSimilarity:
    """
    Класс для представления информации о схожести кадров.

    Attributes:
        frame_index (int): Индекс оригинального кадра.
        similar_frames (List[Dict[str, Any]]): Список схожих кадров с их метаданными.

    Этот класс используется для хранения информации о схожести между
    оригинальным кадром и набором других кадров. Каждый схожий кадр
    представлен словарем с метаданными, такими как идентификатор видео,
    индекс кадра и мера схожести (дистанция).
    """

    def __init__(self, frame_index: int, similar_frames: List[Dict[str, Any]]):
        self.frame_index = frame_index
        self.similar_frames = similar_frames


class VideoSimilarityAnalysis:
    def __init__(self):
        """
        Класс расчёта статистик.

        Attributes:
            frame_similarities (List[FrameSimilarity]): List of FrameSimilarity objects
            video_similarities (Dict[str, List[Dict[str, Any]]]): Dictionary of video_id to list of similar frames
        """
        self.frame_similarities: List[FrameSimilarity] = []
        self.video_similarities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    def add_frame_similarity(self, frame_index: int,
                             similar_frames: List[Dict[str, Any]]):
        """
        Добавляет информацию о схожести кадров в анализ.

        Args:
            frame_index (int): Индекс оригинального кадра.
            similar_frames (List[Dict[str, Any]]): Список схожих кадров с их метаданными.

        Этот метод выполняет следующие действия:
        1. Добавляет новый объект FrameSimilarity в список frame_similarities.
        2. Обновляет словарь video_similarities, добавляя информацию о схожести
           для каждого видео, к которому относятся схожие кадры.

        Каждая запись в video_similarities содержит:
        - 'original_frame': индекс оригинального кадра
        - 'similar_frame': индекс схожего кадра
        - 'distance': расстояние (мера схожести) между кадрами
        """
        self.frame_similarities.append(FrameSimilarity(frame_index, similar_frames))
        for frame in similar_frames:
            self.video_similarities[frame['entity']['video_id']].append({
                'original_frame': int(frame_index),
                'similar_frame': int(frame['entity']['frame_index']),
                'distance': frame['distance']
            })

    def analyze_similarities(self):
        """
        Анализирует схожести между кадрами видео и возвращает результаты анализа.

        Returns:
            Dict[str, Dict[str, Any]]: Словарь с результатами анализа для каждого видео.
            Ключи - идентификаторы видео, значения - словари со следующими полями:
            - 'total_similar_frames': общее количество схожих кадров
            - 'average_distance': средняя дистанция (мера схожести) между кадрами
            - 'frame_sequence': последовательность индексов схожих кадров без последовательных дубликатов
            - 'is_sequential': булево значение, указывающее, является ли последовательность кадров преимущественно возрастающей
            - 'sequence_correlation': корреляция последовательности кадров с идеальной возрастающей последовательностью

        Метод выполняет следующие действия:
        1. Для каждого видео сортирует схожие кадры по индексу оригинального кадра.
        2. Удаляет последовательные дубликаты из последовательности индексов схожих кадров.
        3. Вычисляет статистики на основе схожих кадров.
        4. Определяет, является ли последовательность кадров преимущественно возрастающей.
        5. Рассчитывает корреляцию последовательности кадров с идеальной возрастающей последовательностью.
        """
        results = {}
        for video_id, similarities in self.video_similarities.items():
            similarities.sort(key=lambda x: int(x['original_frame']))
            frame_indices = remove_consecutive_duplicates([sim['similar_frame'] for sim in similarities])

            results[video_id] = {
                'total_similar_frames': len(similarities),
                'average_distance': statistics.mean(
                    sim['distance'] for sim in similarities),
                'frame_sequence': frame_indices,
                'is_sequential': self.is_sequence_mostly_increasing(frame_indices),
                'sequence_correlation': self.calculate_sequence_correlation(
                    frame_indices)
            }
        return results

    @staticmethod
    def is_sequence_mostly_increasing(sequence):
        """
        Определяет, является ли последовательность преимущественно возрастающей.

        Args:
            sequence (List[int]): Последовательность чисел для анализа.

        Returns:
            bool: True, если более половины элементов последовательности возрастают,
                  False в противном случае.

        Метод подсчитывает количество случаев, когда текущий элемент больше предыдущего,
        и сравнивает это количество с половиной длины последовательности.
        """
        increases = sum(
            1 for i in range(1, len(sequence)) if sequence[i] > sequence[i - 1])
        return increases > len(sequence) / 2

    @staticmethod
    def calculate_sequence_correlation(sequence):
        """
        Вычисляет корреляцию между последовательностью и идеальной возрастающей последовательностью.

        Args:
            sequence (List[int]): Последовательность чисел для анализа.

        Returns:
            float: Значение корреляции от -1 до 1.
                   1.0 означает идеальную положительную корреляцию,
                   0.0 - отсутствие корреляции,
                   -1.0 - идеальную отрицательную корреляцию.

        Примечания:
            - Если длина последовательности меньше 2, возвращается 1.0.
            - Метод использует коэффициент корреляции Пирсона.
            - Идеальная возрастающая последовательность генерируется как range(len(sequence)).
        """
        if len(sequence) < 2:
            return 1.0
        increasing_list = list(range(len(sequence)))
        correlation = np.corrcoef(increasing_list, sequence)[
            0, 1]
        return float(correlation)


def add_video(video_path: pathlib.Path, hash_method: ImageHasher, frame_interval: float = EXTRACTION_FPS_RATE) -> List[str]:
    video_id = video_path.stem
    video_dir = video_path.parent
    frames_dir = video_dir / FRAME_DIR
    frames_dir.mkdir(exist_ok=True)

    # раскладываем видео на кадры
    (
        ffmpeg
        .input(str(video_path))
        .filter('fps', fps=EXTRACTION_FPS_RATE)
        .output(str(video_path.parent / 'frames' / '%d.jpg'), start_number=0)
        .global_args('-loglevel', 'error')
        .run()
    )


    # Вычисление хешей для каждого кадра
    for frame_file in sorted(frames_dir.glob('*.jpg'),
                    key=lambda path: int(path.stem)):
        if frame_file.suffix == '.jpg':
            file_index = frame_file.stem
            pil_img = Image.open(frame_file)
            img_hash_original = hash_method.encode(pil_img)

            # Отображение по вертикали
            pil_img_flipped = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            img_hash_flipped = hash_method.encode(pil_img_flipped)

            # Инвертирование каналов RGB
            pil_img_inverted = Image.eval(pil_img, lambda x: 255 - x)
            img_hash_inverted = hash_method.encode(pil_img_inverted)


            # FIXME мы не добавляем видео в поиск
            for h in (img_hash_original, img_hash_flipped, img_hash_inverted):
                db.add_frame_hash(h, file_index, str(video_id))

    return video_id


def process_video(video_path: pathlib.Path, hash_method: ImageHasher, frame_interval: float = EXTRACTION_FPS_RATE) -> List[str]:
    video_id = video_path.stem
    video_dir = video_path.parent
    frames_dir = video_dir / FRAME_DIR
    frames_dir.mkdir(exist_ok=True)

    # раскладываем видео на кадры
    (
        ffmpeg
        .input(str(video_path))
        .filter('fps', fps=EXTRACTION_FPS_RATE)
        .output(str(video_path.parent / 'frames' / '%d.jpg'), start_number=0)
        .global_args('-loglevel', 'error')
        .run()
    )

    # Пример использования
    analysis = VideoSimilarityAnalysis()

    # Временное хранение хэшей
    hashes = []

    # Вычисление хешей для каждого кадра
    for frame_file in sorted(frames_dir.glob('*.jpg'),
                    key=lambda path: int(path.stem)):
        if frame_file.suffix == '.jpg':
            file_index = frame_file.stem
            hash_original_path = frame_file.with_name(f"{file_index}_original.npy")
            hash_flipped_path = frame_file.with_name(f"{file_index}_flipped.npy")
            hash_inverted_path = frame_file.with_name(f"{file_index}_inverted.npy")

            # Оригинальное изображение
            if not hash_original_path.exists():
                pil_img = Image.open(frame_file)
                img_hash_original = hash_method.encode(pil_img)
                np.save(hash_original_path, img_hash_original)
            else:
                img_hash_original = np.load(hash_original_path)

            # Отображение по вертикали
            if not hash_flipped_path.exists():
                pil_img = Image.open(frame_file)
                pil_img_flipped = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                img_hash_flipped = hash_method.encode(pil_img_flipped)
                np.save(hash_flipped_path, img_hash_flipped)
            else:
                img_hash_flipped = np.load(hash_flipped_path)

            # Инвертирование каналов RGB
            if not hash_inverted_path.exists():
                pil_img = Image.open(frame_file)
                pil_img_inverted = Image.eval(pil_img, lambda x: 255 - x)
                img_hash_inverted = hash_method.encode(pil_img_inverted)
                np.save(hash_inverted_path, img_hash_inverted)
            else:
                img_hash_inverted = np.load(hash_inverted_path)

            # db.add_frame_hash(img_hash, file_index, str(video_id))
            for img_hash in (img_hash_original, img_hash_flipped, img_hash_inverted):
                similar_frames = db.search_similar_frames(
                    img_hash,
                    limit=1,
                    filter_expr=f'video_id != "{video_id}"'
                    )
                best_result_list = sorted(
                    [x for x in similar_frames if x['distance'] > 0.75],
                    key=lambda x: x['distance'],
                    reverse=True
                )
                if best_result_list:

                    analysis.add_frame_similarity(file_index, [best_result_list[0]])
                hashes.append({
                    'file_index': file_index,
                    'img_hash': img_hash,
                    'video_id': video_id
                })
    
    # удаляем временные файлы
    try:
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(e)
    results = analysis.analyze_similarities()

    # Проверяем, не пустой ли результат
    if results:
        # Converting to DataFrame
        df = pd.DataFrame.from_dict(results, orient='index')

        # Проверяем наличие необходимых столбцов
        if 'is_sequential' in df.columns and 'total_similar_frames' in df.columns:
            # Sorting by is_sequential first and then by total_similar_frames
            sorted_df = df.sort_values(by=['is_sequential', 'total_similar_frames'],
                                       ascending=[False, False])

            # Применяем фильтры
            result = sorted_df[
                (sorted_df['is_sequential'] == True) &
                (sorted_df['average_distance'] > 0.8) &
                (sorted_df['sequence_correlation'] > 0.9) &
                (sorted_df['frame_sequence'].apply(len) > 20)
                ]
            print()
        else:
            result = pd.DataFrame()  # Пустой DataFrame, если нет нужных столбцов
    else:
        result = pd.DataFrame()  # Пустой DataFrame, если results пустой


    # Вывод индексов в виде массива
    index_array = result.index.tolist()
    if not index_array: # это не дубль
        for img_record in hashes:
            file_index = img_record['file_index']
            img_hash = img_record['img_hash']
            video_id = img_record['video_id']
            db.add_frame_hash(img_hash, file_index, str(video_id))
    return index_array


@app.post("/check-video-duplicate", response_model=VideoLinkResponse, tags=["API для проверки дубликатов видео"])
def check_video_duplicate(video_request: VideoLinkRequest):
    """
    Проверка видео на дублирование

    Args:
    - video_link (VideoLinkRequest): Объект с ссылкой на видео

    Returns:
    - VideoLinkResponse: Результат проверки на дублирование
    """
    try:

        # Скачиваем видео
        video_path = download_video(str(video_request.link))

        if video_path.exists():
            duplicate_indexes = process_video(video_path, hash_method=hasher, frame_interval=EXTRACTION_FPS_RATE)
            if duplicate_indexes:
                return VideoLinkResponse(is_duplicate=True, duplicate_for=duplicate_indexes[0])
            else:
                return VideoLinkResponse(is_duplicate=False, duplicate_for=None)

        # Здесь должна быть реальная логика проверки дубликатов
        # Для примера, мы всегда возвращаем, что видео не является дубликатом
        return VideoLinkResponse(is_duplicate=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка сервера")



@app.post("/add-video", response_model=VideoLinkResponse, tags=["API для добавления роликов видео"])
def add_video_to_db(video_request: VideoLinkRequest):
    """
    Добавляет видео в базу данных без предварительной проверки на дубликаты.

    Этот метод выполняет следующие действия:
    1. Скачивает видео по предоставленной ссылке.
    2. Если видео успешно скачано, извлекает из него кадры и добавляет их хеши в базу данных.
    3. Возвращает результат операции в виде объекта VideoLinkResponse.

    Args:
    - video_request (VideoLinkRequest): Объект, содержащий ссылку на видео для добавления.

    Returns:
    - VideoLinkResponse: Объект с результатом операции.
      - is_duplicate (bool): Всегда True, если видео успешно добавлено.
      - duplicate_for (List[str] или None): Список индексов добавленных кадров или None, если видео не было добавлено.

    Raises:
    - HTTPException: Возникает при ошибке сервера с кодом статуса 500.

    Примечание:
    - Метод не проверяет наличие дубликатов перед добавлением видео.
    - Если видео не удалось скачать, возвращается VideoLinkResponse с is_duplicate=False.
    """
    try:
        # Скачиваем видео
        video_path = download_video(str(video_request.link))

        if video_path.exists():
            duplicate_indexes = add_video(video_path, hash_method=hasher, frame_interval=EXTRACTION_FPS_RATE)
            return VideoLinkResponse(is_duplicate=True, duplicate_for=duplicate_indexes)

        return VideoLinkResponse(is_duplicate=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Ошибка сервера")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
