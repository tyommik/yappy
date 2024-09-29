# Video Duplicate Detection Service

Этот проект представляет собой сервис для обнаружения дубликатов видео с использованием PyTorch и CUDA для ускорения обработки на GPU.

## Требования

- Docker
- NVIDIA GPU с поддержкой CUDA
- NVIDIA Container Toolkit

## Установка

1. Клонируйте репозиторий:
   ```
   git clone https://github.com/tyommik/yappy.git
   cd yappy
   ```

2. Убедитесь, что у вас установлены Docker и NVIDIA Container Toolkit.

3. Соберите Docker образ:
   ```
   docker build -t video-duplicate-detector .
   ```

## Запуск

Запустите контейнер с поддержкой GPU:

```
docker run --gpus all -p 8000:8000 video-duplicate-detector
```

Сервис будет доступен по адресу `http://localhost:8000`.

## Использование

Отправьте POST запрос на `/add-video` с JSON-телом, содержащим ссылку на видео:

```json
{
  "link": "http://example.com/path/to/video.mp4"
}
```

Сервис вернет JSON-ответ с информацией о том, является ли видео дубликатом:

```json
{
  "is_duplicate": true,
  "duplicate_for": "uuid-of-original-video"
}
```

## Структура проекта

- `service.py`: Основной файл сервиса
- `requirements_linux.txt`: Зависимости Python
- `Dockerfile`: Инструкции для сборки Docker-образа

## Лицензия
MIT

## Контакты

[Артём Шибаев] - [artem@shibaev.ru]
[Алексей Дорошенко] - [lehador1@yandex.ru]

Ссылка на проект: https://github.com/tyommik/yappy
