import os
import json
import joblib
import shutil
import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

SEED = 42
warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def create_sequences(data, seq_length, index):
    X, y, timestamps = [], [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
        timestamps.append(index[i + seq_length])
    return np.array(X), np.array(y), np.array(timestamps)


def inverse_transform_predictions(predictions, scaler):
    # make a copy of the DataFrame inside the function to avoid modifying the original:
    predictions = predictions.copy()

    # min/max scaling reversing
    predictions['actual'] = scaler.inverse_transform(predictions['actual'].values.reshape(-1, 1))
    predictions['forecast'] = scaler.inverse_transform(predictions['forecast'].values.reshape(-1, 1))

    # log transformation reversing
    predictions['actual'] = np.exp(predictions['actual']) - 1
    predictions['forecast'] = np.exp(predictions['forecast']) - 1

    return predictions.round(0).astype(int)


# Определим метрику MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Определим RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Определим WAPE
def weighted_absolute_percentage_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# Определим SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    """
    # Защита от деления на ноль (если y_true и y_pred оба нули)
    mask = (np.abs(y_true) + np.abs(y_pred)) != 0
    y_true = np.where(mask, y_true, 1e-6)  # заменяем нули на маленькое число
    y_pred = np.where(mask, y_pred, 1e-6)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


def get_preprocessed_item_data(data_path: str, item_id: int, smoothing_window: int = 10) -> pd.DataFrame:
    # чтение данных
    data = pd.read_csv(data_path, delimiter=';', encoding='cp1251', parse_dates=['Дата время заказа'])
    data = data.sort_values(by='Дата время заказа')

    # предобработка
    item_data = data[data['Артикул товара'] == item_id]
    item_data['Дата время заказа'] = pd.to_datetime(item_data['Дата время заказа'], format='%d.%m.%Y %H:%M:%S')

    # агреггируем по неделям
    item_data['week_start'] = item_data['Дата время заказа'].dt.to_period('W')
    item_data_weekly = item_data.groupby('week_start')['Количество'].sum()

    # в данных должны быть все недели (выравниваем)
    full_index_weekly = pd.period_range(
        start=item_data_weekly.index.min(),
        end=item_data_weekly.index.max(),
        freq='W'
    )

    # реиндексируем, чтобы иметь все дни/недели/месяца
    item_data_weekly = item_data_weekly.reindex(full_index_weekly, fill_value=0)
    item_data_weekly.index = item_data_weekly.index.to_timestamp()

    # логарифмирование таргета -> для лучшего прогнозирования
    item_data_weekly_log = item_data_weekly.apply(lambda x: np.log(x + 1))

    # сглаживаем -> для лучшего прогнозирования
    item_data_weekly_log_smoothed = item_data_weekly_log.rolling(window=smoothing_window, center=False).mean()

    # удаляем пропуски
    item_data_weekly_log_smoothed = item_data_weekly_log_smoothed.dropna()
    return item_data_weekly_log_smoothed


def get_training_data(item_data: pd.DataFrame, scaler: any, seq_len: int, weeks_test: int) -> tuple:
    data_scaled = scaler.fit_transform(item_data.values.reshape(-1, 1))

    X, y, y_timestamps = create_sequences(data_scaled, seq_len, item_data.index)
    train_size = len(X) - weeks_test

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    time_train, time_test = y_timestamps[:train_size], y_timestamps[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, X_test, y_train, y_test, time_train, time_test


def get_lstm_model(seq_len: int = 15, **kwargs) -> Any:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_len, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(**kwargs)
    return model


def train_lstm_model(model: Any, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
    history = model.fit(X_train, y_train, **kwargs)


def validate_lstm_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    forecast_horizon: int = 10
):
    # предобработка
    X_test_horizon = X_test[:forecast_horizon]
    y_test_horizon = y_test[:forecast_horizon]

    # прогнозируем
    model_preds = model.predict(X_test_horizon)

    # метрики
    smape = symmetric_mean_absolute_percentage_error(y_test_horizon, model_preds)
    wape = root_mean_squared_error(y_test_horizon, model_preds)
    rmse = weighted_absolute_percentage_error(y_test_horizon, model_preds)

    return {
        'smape': smape,
        'wape': wape,
        'rmse': rmse,
    }


def predict(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_test: np.ndarray,
    forecast_horizon: int = 10
):
    # предобработка
    X_test_horizon = X_test[:forecast_horizon]
    y_test_horizon = y_test[:forecast_horizon]
    time_test_horizon = time_test[:forecast_horizon]

    # прогнозируем
    model_preds = model.predict(X_test_horizon)

    forecast_df = pd.DataFrame({
        'timestamp': time_test_horizon,
        'actual': y_test_horizon.flatten(),
        'forecast': model_preds.flatten()
    }).set_index('timestamp')

    return forecast_df


def show_model_forecast(
    forecast_df: pd.DataFrame,
    title: str,
    how: str = 'local',
    y_train: np.ndarray = None,
    time_train: np.ndarray = None
):
    # визуализируем ряд
    fig = go.Figure()

    if how == 'local':
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df['actual'],
                mode='lines',
                name='Исторические данные',
                line=dict(color='blue')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'],
                mode='lines',
                name='Прогноз',
                line=dict(color='red', dash='dash')
            )
        )

    if how == 'global':
        # обучающая выборка
        fig.add_trace(go.Scatter(
            x=time_train,
            y=y_train.flatten(),  # убедитесь, что y_train имеет нужную форму
            mode='lines',
            name='Исторические данные',
            line=dict(color='blue', width=1.5)
        ))

        # фактические значения
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['actual'],
            mode='lines',
            name='Известные значения',
            line=dict(color='green')
        ))

        # прогноз
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['forecast'],
            mode='lines',
            name='Прогноз',
            line=dict(color='red', dash='dash')
        ))

    # настройки графика
    fig.update_layout(
        title=title,
        xaxis_title="Время",
        yaxis_title="Количество (отмасштабированное)",
        height=600,
        width=1400,
    )

    return fig


def save_model_and_data(
    model,
    scaler,
    X_test, 
    y_test,
    y_train,
    time_train,
    time_test,
    model_path="lstm_model.h5",
    data_path="model_data.npz",
    scaler_path="scaler.save"
):
    """
    Saves a Keras model and associated NumPy arrays to disk.
    """
    # save the model
    model.save(model_path)

    # savel scaler
    joblib.dump(value=scaler, filename=scaler_path)

    # save the data
    np.savez(
        data_path, 
        X_test=X_test, 
        y_test=y_test, 
        y_train=y_train, 
        time_train=time_train,
        time_test=time_test
    )

    print(f"Model saved to {model_path}")
    print(f"Data saved to {data_path}")


def load_model_and_data(
    model_path="lstm_model.h5",
    scaler_path="scaler.save",
    data_path="model_data.npz"
) -> tuple:
    """
    Loads a Keras model and associated NumPy arrays from disk.
    Returns:
        model, X_test, y_test, y_train, time_train
    """
    # load the model
    model = load_model(model_path)

    # load scaler
    scaler = joblib.load(filename=scaler_path)

    # load the data
    data = np.load(data_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_train = data["y_train"]
    time_train = data["time_train"]
    time_test = data["time_test"]

    print(f"Model loaded from {model_path}")
    print(f"Data loaded from {data_path}")

    return model, scaler, X_test, y_test, y_train, time_train, time_test


def get_top_items(data_path: str, top_k: int = 10) -> list:
    data = pd.read_csv(data_path, delimiter=';', encoding='cp1251', parse_dates=['Дата время заказа'])
    data = data.sort_values(by='Дата время заказа')

    top_items = data['Артикул товара'].value_counts().head(top_k).index.tolist()
    return top_items


def save_metrics_as_json(metrics_dict: dict, filename="metrics.json"):
    """
    Saves a dictionary of metrics to a JSON file, converting NumPy types to native Python types.
    
    Args:
        metrics_dict (dict): Dictionary containing metric names and values (possibly NumPy types).
        filename (str): Name of the output JSON file.
    """
    # Convert all NumPy float types to native Python floats
    serializable_metrics = {
        k: float(v) if isinstance(v, np.generic) else v 
        for k, v in metrics_dict.items()
    }
    
    with open(filename, "w") as f:
        json.dump(serializable_metrics, f, indent=4)
    
    print(f"Metrics saved to {filename}")


def train_and_save_items_model(data_path: str, weeks_count_vaidation: int, top_k: int = 10) -> None:
    SEQ_LEN = 15
    WEEKS_TEST = 100
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    VERBOSE = 0
    SMOOTHING_WINDOW = 10

    # here models data per item will be stored
    if os.path.exists('models_data'):
        shutil.rmtree('models_data')
    os.mkdir('models_data')

    top_items = get_top_items(data_path, top_k)

    for item_id in tqdm(top_items):
        print(f'\nProcess Item {item_id}')
        try:
            scaler = MinMaxScaler()
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            item_data = get_preprocessed_item_data(
                data_path=data_path,
                item_id=item_id,
                smoothing_window=SMOOTHING_WINDOW
            )
            print('Preprocess Data -> Done')

            X_train, X_test, y_train, y_test, time_train, time_test = get_training_data(
                item_data=item_data, scaler=scaler, seq_len=SEQ_LEN, weeks_test=WEEKS_TEST
            )

            lstm_model = get_lstm_model(optimizer='adam', loss='mean_squared_error')

            train_lstm_model(
                model=lstm_model,
                X_train=X_train,
                y_train=y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_split=VALIDATION_SPLIT,
                verbose=VERBOSE,
                callbacks=[early_stop]
            )
            print('Train Model -> Done')

            validation_results = validate_lstm_model(
                model=lstm_model,
                X_test=X_test,
                y_test=y_test,
                forecast_horizon=weeks_count_vaidation
            )

            print('Validate Model -> Done')

            # add number of weeks on what model was validated
            validation_results['weeks_count_vaidation'] = weeks_count_vaidation

            save_metrics_as_json(
                metrics_dict=validation_results,
                filename=f'models_data/item_{item_id}_metrics_validation.json'
            )

            save_model_and_data(
                model=lstm_model,
                scaler=scaler,
                X_test=X_test,
                y_test=y_test,
                y_train=y_train,
                time_train=time_train,
                time_test=time_test,
                model_path=f'models_data/item_{item_id}_model.h5',
                data_path=f'models_data/item_{item_id}_model_data.npz',
                scaler_path=f'models_data/item_{item_id}_scaler.save',
            )

            print(f'Item {item_id} -> Done')

        except:
          print(f'Error with item {item_id}')