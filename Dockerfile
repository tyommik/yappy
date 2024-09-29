# Используем образ PyTorch с поддержкой CUDA
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements_linux.txt /app/

# Устанавливаем зависимости
RUN pip install -r requirements_linux.txt

# Копируем весь код из текущей директории в /app в контейнере
COPY . /app/

# Указываем, что контейнер слушает порт 8000
EXPOSE 8000
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Запускаем сервис при старте контейнера
CMD ["python3", "service.py"]