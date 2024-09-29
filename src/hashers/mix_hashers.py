from imagededup.methods import PHash, CNN
import numpy as np
from PIL import Image

from .base_hasher import ImageHasher


class MixHash(ImageHasher):
    def __init__(self):
        self.phasher = PHash()
        self.cnn = CNN()

    def encode(self, pil_image):
        """
        Кодирует PIL изображение, вычисляя его PHash и CNN эмбеддинг,
        затем конкатенирует их.

        :param pil_image: PIL.Image объект
        :return: конкатенированный вектор хеша и эмбеддинга
        """
        # Преобразуем PIL изображение в numpy array
        np_image = np.array(pil_image)

        # Вычисляем хеш изображения
        hash_value = self.phash_to_vector(self.phasher.encode_image(image_array=np_image))
        
        # Вычисляем эмбеддинг изображения
        embedding_value = self.cnn.encode_image(image_array=np_image)

        # Конкатенируем векторы
        concatenated_vector = np.concatenate((hash_value, embedding_value[0]))

        return concatenated_vector

    def encode_file(self, image_path):
        """
        Кодирует изображение из файла, вычисляя его PHash.

        :param image_path: путь к файлу изображения
        :return: хеш изображения
        """
        with Image.open(image_path) as img:
            return self.encode(img)


    @staticmethod
    def phash_to_vector(phash_string):
        # Преобразуем шестнадцатеричную строку в бинарный вектор
        binary_list = [int(char, 16) for char in phash_string]
        binary_vector = []
        for num in binary_list:
            binary_vector.extend([int(bit) for bit in format(num, '04b')])

        # Убедимся, что длина вектора совпадает с ожидаемой (например, 64)
        if len(binary_vector) != 64:
            raise ValueError("Неправильная длина pHash, ожидался 64-битный хэш")

        float_vector = np.array(binary_vector, dtype=np.float32)
        return float_vector