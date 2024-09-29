import os
import pandas as pd
import json

# Чтение исходного CSV файла
df = pd.read_csv('../data/test.csv')

# Список для хранения результатов
results = []

# Проходим по каждой строке в исходном CSV
for index, row in df.iterrows():
    uuid = row['uuid']
    json_path = f"../data/output/{uuid}.json"

    # Проверяем, существует ли JSON-файл
    if os.path.exists(json_path):
        # Чтение JSON файла
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        # Извлекаем необходимые данные из JSON
        is_duplicate = data.get('is_duplicate', False)
        duplicate_for = data.get('duplicate_for', '')
    else:
        # Если JSON файл отсутствует, устанавливаем значения по умолчанию
        is_duplicate = False
        duplicate_for = ''

    # Добавляем строку с результатами в список
    results.append({
        'created': row['created'],
        'uuid': row['uuid'],
        'link': row['link'],
        'is_duplicate': is_duplicate,
        'duplicate_for': duplicate_for
    })

# Преобразуем результаты в DataFrame
submit_df = pd.DataFrame(results)

# Запись в новый CSV файл
submit_df.to_csv('../data/submit_03.csv', index=False)