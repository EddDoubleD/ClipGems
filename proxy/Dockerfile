# Используем официальный образ Python 3.9.6
FROM python:3.9.6-slim

RUN apt-get update && \
    apt-get install -y git

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git


# Копируем исходный код приложения в контейнер
COPY ./src .

# Устанавливаем переменную окружения, которая указывает, что приложение работает в окружении, поддерживающем многопотоковые запросы
ENV PYTHONUNBUFFERED=1

# Указываем команду для запуска нашего приложения
CMD ["python3", "main.py"]