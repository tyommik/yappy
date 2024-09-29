import pandas as pd
import requests
from sklearn.metrics import roc_auc_score
import tqdm

# Чтение CSV файла
df = pd.read_csv('../data/train.csv')
df['is_duplicate'] = df['is_duplicate'].astype(str)
df['is_hard'] = df['is_hard'].astype(str)

# Фильтрация данных
filtered_df = df[(df['is_duplicate'] == 'False') & (df['is_hard'] == 'False')]
filtered_df = filtered_df[:200]

# Функция для отправки запроса и обработки ответа
def send_request(link):
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
for index, row in tqdm.tqdm(filtered_df.iterrows()):
    is_duplicate, duplicate_for = send_request(row['link'])
    if is_duplicate is not None:
        results.append({
            'uuid': row['uuid'],
            'original_is_duplicate': row['is_duplicate'],
            'predicted_is_duplicate': is_duplicate,
            'duplicate_for': duplicate_for
        })

# Создание DataFrame из результатов
results_df = pd.DataFrame(results)

# Сохранение результатов в CSV
results_df.to_csv('prediction_results.csv', index=False)

# Подготовка данных для расчета ROC-AUC
y_true = (results_df['original_is_duplicate'] == 'True').astype(int)
y_pred = results_df['predicted_is_duplicate'].astype(int)

# Расчет ROC-AUC
roc_auc = roc_auc_score(y_true, y_pred)

print(f"ROC-AUC Score: {roc_auc}")

# Анализ результатов
total_predictions = len(results_df)
correct_predictions = sum(results_df['original_is_duplicate'] == results_df['predicted_is_duplicate'].astype(str))
accuracy = correct_predictions / total_predictions

print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}")

# Анализ дубликатов
duplicates = results_df[results_df['predicted_is_duplicate'] == True]
print(f"Number of predicted duplicates: {len(duplicates)}")
print(f"Unique 'duplicate_for' values: {duplicates['duplicate_for'].nunique()}")