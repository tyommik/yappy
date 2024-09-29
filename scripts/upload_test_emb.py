import os
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Чтение CSV файла
df = pd.read_csv('../data/test.csv')[908:]

# Функция для отправки запроса и обработки ответа
def send_request(row):
    link = row['link'].replace('https://s3.ritm.media/yappy-db-duplicates/', 'http://192.168.204.212:8000/output/')
    payload = {"link": link}
    try:
        response = requests.post(
            "http://192.168.204.212:8888/add-video",
            json=payload,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Python/3.x'
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(link, data)
            is_duplicate = data.get('is_duplicate', False)
            duplicate_for = data.get('duplicate_for', None)
            print(row['uuid'], is_duplicate, duplicate_for)
            return {
                'created': row['created'],
                'uuid': row['uuid'],
                'link': row['link'],
                'is_duplicate': is_duplicate,
                'duplicate_for': duplicate_for
            }
        else:
            print(f"Ошибка при загрузке видео {link}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Исключение при обработке {link}: {str(e)}")
        return None

# Функция для многопоточной обработки
def process_videos(df, max_workers=10):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(send_request, row): row for _, row in df.iterrows()}
        for future in tqdm.tqdm(as_completed(future_to_row), total=len(future_to_row)):
            result = future.result()
            if result is not None:
                results.append(result)
    return results

# Запуск многопоточной обработки
results = process_videos(df)

# Создание DataFrame из результатов
results_df = pd.DataFrame(results)

# Сохранение результатов в CSV
# results_df.to_csv('prediction_results.csv', index=False)

print(f"Обработано {len(results)} видео из {len(df)} исходных записей.")