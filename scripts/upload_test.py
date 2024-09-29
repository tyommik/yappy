import os
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
import tqdm
import json

# Чтение CSV файла
df = pd.read_csv('../data/test.csv')[570:]


# Функция для отправки запроса и обработки ответа
def send_request(link, uuid):
    link = link.replace('https://s3.ritm.media/yappy-db-duplicates/', 'http://192.168.204.212:8000/output/')
    payload = {"link": link}
    try:
        response = requests.post(
            "http://192.168.204.212:8000/check-video-duplicate",
            json=payload,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Python/3.x'
            }
        )
        if response.status_code == 200:
            data = response.json()
            print(link, data)

            # Сохранение ответа под именем {uuid}.json в папку data/output
            output_path = f"../data/output/{uuid}.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as outfile:
                json.dump(data, outfile, indent=4)

            return data.get('is_duplicate', False), data.get('duplicate_for', None)
        else:
            print(f"Ошибка при загрузке видео {link}: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Исключение при обработке {link}: {str(e)}")
        return None, None

# Список для хранения результатов
results = []

# Отправка запросов для каждой записи
for index, row in tqdm.tqdm(df.iterrows()):
    is_duplicate, duplicate_for = send_request(row['link'], row['uuid'])
    print(row['uuid'], is_duplicate, duplicate_for)
    if is_duplicate is not None:
        results.append({
            'created': row['created'],
            'uuid': row['uuid'],
            'link': row['link'],
            'is_duplicate': is_duplicate,
            'duplicate_for': duplicate_for
        })

# Создание DataFrame из результатов
results_df = pd.DataFrame(results)

# Сохранение результатов в CSV
results_df.to_csv('prediction_results.csv', index=False)

