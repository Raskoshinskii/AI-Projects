# используем python 3.10
FROM python:3.10-slim

# определяем рабочий каталог в контейнере
WORKDIR /app

# копируем необходимые файлы в рабочий каталог app
COPY ./telegram_bot/ ./
COPY requirements.txt ./

# устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# запускаем скрипт
CMD ["python", "main.py"] 
