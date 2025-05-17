import warnings

from preprocessing import (
    train_and_save_items_model,
    load_model_and_data,
    predict,
    show_model_forecast,
    show_model_forecast
)

warnings.filterwarnings('ignore')


# обучает модели на артикул товара и сохраняет необходимые данные по ним
train_and_save_items_model(
    data_path='data/orders_new.csv',   # данные с файлом
    weeks_count_vaidation=100,         # на скольких неделях валидировать модель
    top_k=10
)