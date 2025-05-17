import os
# import telebot
import logging

# from telebot import types
# from dotenv import load_dotenv


# конфигурируем logging / INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# # подгружаем перменные среды
# load_dotenv()


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


# @bot.message_handler(commands=['start'])
# def start_message(message):
#     # создаем клавиатуру
#     keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    
#     # создаем кнопки
#     button_graph = types.KeyboardButton(GRAPH_BUTTON_TEXT)
#     button_interest_rate = types.KeyboardButton(INTEREST_RATE_BUTTON_TEXT)
    
#     # добавляем кнопки на клавиатуру
#     keyboard.add(button_graph, button_interest_rate)
    
#     # отправляем приветственное сообщение с клавиатурой
#     bot.send_message(message.chat.id, START_MESSAGE, reply_markup=keyboard)


# @bot.message_handler(func=lambda message: True)
# def handle_message(message):
#     if message.text == GRAPH_BUTTON_TEXT:
#         # TODO: добавить логику для отображения графика
#         pass

#     if message.text == INTEREST_RATE_BUTTON_TEXT:
#         banks_data = get_interest_rate_data()
        
#         # красиво выводим сообщение по ставкам банков
#         response = INTEREST_RATE_RESPONSE_HEADER
#         for bank_data in banks_data:
#             response += INTEREST_RATE_RESPONSE_STRUCTURE.format(**bank_data)

#         response += INTEREST_RATE_RESPONSE_FOOTER

#         # отправляем финальное сообщение
#         bot.send_message(message.chat.id, response)
#         logging.info('Successfully sent message | RATE')


