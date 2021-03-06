### Прогнозирование Числа Поездок Такси в Нью-Йорке 
### Описание Проекта 
В данном проекте решается задача прогнозирования спроса (число поездок для желтого такси в определенном районе Нью-Йорка)
Прогнозировать необходимо на ближайший час/часы, следовательно имеем дело с временными рядами. 

Задачи прогнозирования временных рядов часто возникают на практике: 
- Прогноз большого количества товаров в большом количестве магазинов
- Объём снятия денег в сети банкоматов
- Посещаемость разных страниц сайта
- ...

Сырые <a href='https://github.com/vadikl/AI-Projects/tree/main/Machine_Learning/RUS/TeleCom_Churn_Prediction'>данные</a> были агрегированы по часам и районам 
города за период `2020-01` по `2021-07`. Так как данных очень много, то была выдвинута идея кластеризации временных рядов 
и прогнозирование не для отдельного ряда, а для целевого ряда кластера.

### Цель Проекта 
Уметь предсказывать количество поездок в ближайшие часы в каждом районе Нью-Йорка.

### Задачи Проекта
1) Сбор Данных
2) Первичный анализ и предобработка данных
3) Создание ETL Pipeline для получения "чистых" и необходимых данных
4) Определение метода кластеризации временных рядов 
5) Определение метода для извлечения целевого ряда в кластере 
6) Выбор регрессионной модели
7) Генерация регрессионных признаков 
8) Валидация полученных результатов 

### Метрики 
В проекте имеем несколько задач: кластеризация (промежуточная/вспомогательная) и регрессия (основная)
- Кластеризация: `Inertia (Elbow Method)` и `Silhouette`
- Регрессия: `MAPE/MAE`

### Модели 
- Кластеризация: `TimeSeriesKMeans/HAC` на признаках меньшей размерности `t-SNE/PCA/MDS/Time Series Embeddings` 
- Регрессия: `LassoCV/LinearRegression/XGBoost`
