### New York Taxi Trips Prediction

### Project Description
This project solves the problem of forecasting demand (the number of trips for a yellow taxi in a certain New York district)
It is necessary to predict for the next hour/hours, therefore we are dealing with time series.

Time series forecasting problems often arise in practice:
- Forecast of a large number of products in a large number of stores
- The amount of money withdrawn from the ATM network
- Attendance of different pages of the site
- ...

Raw <a href='https://github.com/vadikl/AI-Projects/tree/main/Machine_Learning/RUS/TeleCom_Churn_Prediction'>data</a> was aggregated by hours and districts
for the period `2020-01` to `2021-07`. Since there is a lot of data, the idea of time series clustering was put forward.
Forecasting not for a separate series, but for the target series of the cluster.

### Project Objective
Be able to predict the number of trips in the coming hours in each New York district

### Project Tasks
1) Data collection
2) Initial data analysis and preprocessing
3) Creating an ETL Pipeline to get "clean" and required data
4) Definition of time series clustering method
5) Defining a method to retrieve the target series in the cluster
6) Regression model selection
7) Regression features generation
8) Obtained results validation

### Metrics
We have several tasks in the project: clustering (intermediate/auxiliary) and regression (main)
- Clustering: `Inertia (Elbow Method)` and `Silhouette`
- Regression: `MAPE/MAE`

### Models
- Clustering: `TimeSeriesKMeans/HAC` on lesser features `t-SNE/PCA/MDS/Time Series Embeddings`
- Regression: `LassoCV/LinearRegression/XGBoost`
