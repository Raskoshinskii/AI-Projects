import os
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from dotenv import load_dotenv

from .constants import (
    INTEREST_RATE_QUERY,
    INTEREST_RATE_RESPONSE_STRUCTURE,
    INTEREST_RATE_RESPONSE_HEADER,
    INTEREST_RATE_RESPONSE_FOOTER
)

from .utils import is_docker_env

# загружаем переменные среды
load_dotenv()

# определяем параметры подлчения к postgres
user_name = os.getenv("POSTGRES_DB_USER")
pwd = os.getenv("POSTGRES_DB_PASSWORD")
db_name = os.getenv("POSTGRES_DB_NAME")
port = os.getenv("POSTGRES_DB_HOST_PORT")
host = 'localhost'

# название хоста будет другим если запускаем в Docker
if is_docker_env():
    host = 'postgres_db'
    port = os.getenv("POSTGRES_DB_CONTAINER_PORT")


def get_interest_rate_data():
    # создаем подключение к postgres
    engine = create_engine(f'postgresql://{user_name}:{pwd}@{host}:{port}/{db_name}')

    with engine.connect() as connection:
        result = connection.execute(text(INTEREST_RATE_QUERY))
        rows = result.fetchall()
        columns = result.keys()

    return [dict(zip(columns, row)) for row in rows]


def get_message():
    """TODO"""
    
    # получаем данные из БД
    banks_data = get_interest_rate_data()
        
    # формируем красивый ответ/response по ставкам банков
    response = INTEREST_RATE_RESPONSE_HEADER
    for bank_data in banks_data:
        response += INTEREST_RATE_RESPONSE_STRUCTURE.format(**bank_data)

    response += INTEREST_RATE_RESPONSE_FOOTER
    return response