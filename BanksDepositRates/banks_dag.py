import os
import time
import pandas as pd
import requests
import logging
import telebot

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from data_scraper.constants import (
    MAIN_URL,
    QUERY_PARAMS,
    BANK_DICT,
    OFFERS_DICT
)

from data_scraper.main import get_online_rate
from telegram_bot.db import get_message

load_dotenv()


def get_data_from_sravni(ti):
    logging.info('Make Request ...')

    response = requests.post(MAIN_URL, json=QUERY_PARAMS)
    data = response.json()

    logging.info('Get JSON -> Success!')

    ti.xcom_push(key='raw_data', value=data)
    logging.info('Raw data pushed to XCom.')


def preprocess_data(ti):        
    logging.info('Fetching raw data from XCom...')

    # Pull data from XCom
    data = ti.xcom_pull(key='raw_data', task_ids='get_data_from_sravni')

    logging.info('Parsing JSON ...')

    # Parse response
    for item in data["items"]:
        BANK_DICT["name"].append(item["organization"]["name"]["short"])
        BANK_DICT["rate"].append(item["rate"])
        BANK_DICT["online_rate"].append(get_online_rate(item))
        BANK_DICT["term"].append(item["term"])
        BANK_DICT["amount_from"].append(item["amount"]["from"])
        BANK_DICT["amount_to"].append(item["amount"]["to"])
        BANK_DICT["offer_count"].append(item["groupCount"])

        if item["groupCount"] > 0:
            for group_item in item['groupItems']:
                OFFERS_DICT["bank_name"].append(item["organization"]["name"]["short"])
                OFFERS_DICT["rate"].append(group_item["rate"])
                OFFERS_DICT["online_rate"].append(get_online_rate(group_item))
                OFFERS_DICT["term"].append(group_item["term"])
                OFFERS_DICT["amount_from"].append(group_item["amount"]["from"])
                OFFERS_DICT["amount_to"].append(group_item["amount"]["to"])

    logging.info('Parse JSON -> Success!')

    # Create DataFrames
    bank_df = pd.DataFrame(BANK_DICT)
    offers_df = pd.DataFrame(OFFERS_DICT)

    logging.info('Preprocess DataFrames -> ...')

     # Process data
    bank_df['rate'] = bank_df['rate'].apply(lambda x: x.strip('до% '))
    bank_df['rate'] = bank_df['rate'].astype('float')
    bank_df['final_rate'] = bank_df[['rate', 'online_rate']].max(axis=1)

    offers_df['rate'] = offers_df['rate'].apply(lambda x: x.strip(' до%'))
    offers_df['rate'] = offers_df['rate'].astype('float')
    offers_df['final_rate'] = offers_df[['rate', 'online_rate']].max(axis=1)

    # Add date
    bank_df['date'] = pd.Timestamp.today()
    offers_df['date'] = pd.Timestamp.today()

    logging.info('Preprocess DataFrames -> Success!')

    # Convert to JSON
    bank_json = bank_df.to_json(orient='records')
    offers_json = offers_df.to_json(orient='records')

    # to xcom
    ti.xcom_push(key='bank_data', value=bank_json)
    ti.xcom_push(key='offers_data', value=offers_json)


def send_data_to_postgres(ti):
    logging.info('Fetching JSONs from XCom...')

    bank_json = ti.xcom_pull(key='bank_data', task_ids='preprocess_data')
    offers_json = ti.xcom_pull(key='offers_data', task_ids='preprocess_data')
    bank_df = pd.read_json(bank_json, orient='records')
    offers_df = pd.read_json(offers_json, orient='records')

    logging.info('JSON to DataFrames -> Success!')

    # Postgres Credentials
    user_name = os.getenv("POSTGRES_DB_USER")
    pwd = os.getenv("POSTGRES_DB_PASSWORD")
    port = os.getenv("POSTGRES_DB_CONTAINER_PORT")
    db_name = os.getenv("POSTGRES_DB_NAME")
    host = 'postgres_db'

    engine = create_engine(f'postgresql://{user_name}:{pwd}@{host}:{port}/{db_name}')

    # TODO: fix -> not goo implemenation
    try:
        bank_df.to_sql(name='banks', con=engine, if_exists='append', index=False)
        offers_df.to_sql(name='offers', con=engine, if_exists='append', index=False)
    except:
        bank_df.to_sql(name='banks', con=engine, if_exists='replace', index=False)
        offers_df.to_sql(name='offers', con=engine, if_exists='replace', index=False)

    logging.info('DataFrames to Postgres -> Success!')


def send_message_to_telegram():
    # определяем бота
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    VLAD_CHAT_ID = os.getenv("VLAD_CHAT_ID")
    ALEX_CHAT_ID = os.getenv("ALEX_CHAT_ID")

    bot = telebot.TeleBot(TELEGRAM_TOKEN)

    # формируем сообщение
    message = get_message()

    # отправляем сообщения
    bot.send_message(VLAD_CHAT_ID, message)
    bot.send_message(ALEX_CHAT_ID, message)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'start_date': datetime(2025, 1, 11),
    'retries': 1,  # Retry once in case of failure
}

with DAG(
    'get_banks_data_sravni',
    default_args=default_args,
    description='A DAG to parse and preprocess banks data from Sravni using XCom',
    schedule_interval='@daily',
    catchup=False,
    tags=['example'],
) as dag:

    # Task 1: Parse data
    parse_data_task = PythonOperator(
        task_id='get_data_from_sravni',
        python_callable=get_data_from_sravni,
    )

    # Task 2: Preprocess data
    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    # Task 3: Save Data to Postgres
    df_to_postgres_task = PythonOperator(
        task_id='send_data_to_postgres',
        python_callable=send_data_to_postgres,
    )

    # Task 4: Send to Telegram Bot
    send_data_to_tg_bot_task = PythonOperator(
        task_id='send_message_to_telegram',
        python_callable=send_message_to_telegram,
    )


    # Task dependencies
    parse_data_task >> preprocess_data_task >> df_to_postgres_task >> send_data_to_tg_bot_task
