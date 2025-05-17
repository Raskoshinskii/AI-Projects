# Базовый образ с Python
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование всех файлов проекта в контейнер
COPY . .

# Установка Python-зависимостей
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Установка Streamlit (если не в requirements.txt)
RUN pip install streamlit

# Указание команды запуска
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
