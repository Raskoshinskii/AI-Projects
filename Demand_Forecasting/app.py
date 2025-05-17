import os
import warnings
import streamlit as st

from preprocessing import (
    predict,
    load_model_and_data,
    show_model_forecast,
    inverse_transform_predictions
)

warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
st.title("📈 Прогноз продаж по товару")

item_id = st.text_input("Введите ID товара:", "505")

if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'forecast_horizon' not in st.session_state:
    st.session_state.forecast_horizon = 10

if st.button("🔍 Загрузить и спрогнозировать"):
    base_path = "models_data"
    model_path = f"{base_path}/item_{item_id}_model.h5"
    data_path = f"{base_path}/item_{item_id}_model_data.npz"
    scaler_path = f"{base_path}/item_{item_id}_scaler.save"

    if not all(os.path.exists(p) for p in [model_path, data_path, scaler_path]):
        st.error("Файлы модели не найдены.")
    else:
        try:
            model, scaler, X_test, y_test, y_train, time_train, time_test = load_model_and_data(
                model_path=model_path,
                data_path=data_path,
                scaler_path=scaler_path,
            )
            st.session_state.model_data = {
                "model": model,
                "scaler": scaler,
                "X_test": X_test,
                "y_test": y_test,
                "y_train": y_train,
                "time_train": time_train,
                "time_test": time_test,
                "scaler": scaler
            }
            st.session_state.forecast_horizon = 10
            st.session_state.forecast_df = None
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

if st.session_state.model_data is not None:
    # Отображаем слайдер, чтобы выбрать горизонт
    forecast_horizon = st.slider(
        "Горизонт прогноза (недель):",
        10, 101,
        value=st.session_state.forecast_horizon
    )

    # Если изменилось значение слайдера — обновляем в session_state и сбрасываем прогноз
    if forecast_horizon != st.session_state.forecast_horizon:
        st.session_state.forecast_horizon = forecast_horizon
        st.session_state.forecast_df = None

    # Если прогноз ещё не посчитан — считаем
    if st.session_state.forecast_df is None:
        try:
            data = st.session_state.model_data
            st.session_state.forecast_df = predict(
                model=data["model"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                time_test=data["time_test"],
                forecast_horizon=st.session_state.forecast_horizon
            )
        except Exception as e:
            st.error(f"Ошибка при прогнозе: {str(e)}")
            st.stop()  # Останавливаем дальнейшую отрисовку

    # Отрисовываем графики и кнопку скачивания
    forecast_df = st.session_state.forecast_df
    data = st.session_state.model_data

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            show_model_forecast(
                forecast_df=forecast_df,
                how='local',
                title='Прогноз модели',
                y_train=data["y_train"],
                time_train=data["time_train"]
            ),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            show_model_forecast(
                forecast_df=forecast_df,
                how='global',
                title='Прогноз модели',
                y_train=data["y_train"],
                time_train=data["time_train"]
            ),
            use_container_width=True
        )

    # инверитруем предсказания
    forecast_df = inverse_transform_predictions(predictions=forecast_df, scaler=data['scaler'])
    csv = forecast_df.reset_index().to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Скачать предсказания (CSV)",
        data=csv,
        file_name=f"forecast_item_{item_id}.csv",
        mime='text/csv'
    )
