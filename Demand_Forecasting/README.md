# How to Run?
Локально:
- python -m env my_env
- source my_env/bin/activate
- pip install -r requirements.txt
- Обучаем модели: python get_models.py

Докер:
- docker build -t streamlit-forecast-app .
- docker run -p 8501:8501 streamlit-forecast-app