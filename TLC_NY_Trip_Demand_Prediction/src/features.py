import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.decomposition import PCA 


def sin_transform(values):
    """
    Takes into account cyclic nature of a feature

    values: pd.DataFrame or pd.Series
        Values to get sin harmonics for 
    Returns:
    -------
    np.array with sin harmonics 
    """
    return np.sin(2*np.pi*values/len(set(values)))


def cos_transform(values):
    """
    values: pd.DataFrame or pd.Series
        Values to get sin harmonics for 
    Returns:
    -------
    np.array with cos harmonics 
    """
    return np.cos(2*np.pi*values/len(set(values)))


def get_time_features(series, is_cyclical_encoding=False):
    """
    Note:
    ----
    Some features are commented (uncomment if needed)

    series: pd.Series or pd.DataFrame 
        Time Sereis Data
    is_cyclical_encoding: bool
        Wether to apply sin/cos transformations
    Returns:
    -------
    DataFrame with time features 
    """
    features_df = series.copy() 
    
    features_df['hour'] = features_df.index.hour
    features_df['day_of_month'] = features_df.index.day
#     features_df['week_of_year'] = features_df.index.weekofyear
    
    features_df['day_name'] = features_df.index.day_name()
#     features_df['month_name'] = features_df.index.month_name()
    
    if is_cyclical_encoding:
        features_df['sin_hour'] = sin_transform(features_df['hour'])
        features_df['cos_hour'] = cos_transform(features_df['hour'])
        
        features_df['sin_week_of_year'] = sin_transform(features_df['week_of_year'])
        features_df['cos_week_of_year'] = cos_transform(features_df['week_of_year'])
        features_df.drop(['hour', 'week_of_year'], axis=1, inplace=True)

    features_df['is_weekend'] = features_df.index.weekday.isin([5,6])*1 # Умножаем на 1 чтобы получить 0/1 вместо True/False
    holidays = USFederalHolidayCalendar().holidays(start=features_df.index.min(), end=features_df.index.max()).floor('1D')
    features_df['is_holiday'] = features_df.index.floor('1D').isin(holidays)*1
    
    lunch_time = pd.date_range('12:00:00', '13:00:00', freq='h').time
    working_hours = pd.date_range('8:00:00', '19:00:00', freq='h').time
    working_hours = set(working_hours) - set(lunch_time)

    features_df['is_working_hour'] = (features_df.reset_index()['tpep_pickup_datetime'].apply(lambda x: x.time() in working_hours)*1).values
    features_df['is_lunch_time'] = (features_df.reset_index()['tpep_pickup_datetime'].apply(lambda x: x.time() in lunch_time)*1).values
    
    features_df['is_month_start'] = features_df.index.is_month_start*1
    features_df['is_month_end'] = features_df.index.is_month_end*1
    features_df['is_quarter_start'] = features_df.index.is_quarter_start*1
    features_df['is_quarter_end'] = features_df.index.is_quarter_end*1
#     features_df['is_year_start'] = features_df.index.is_year_start*1
#     features_df['is_year_end'] = features_df.index.is_year_end*1
    
#     seasons = {1:'Winter', 2:'Spring', 3:'Summer', 4:'Autumn'}
#     features_df['season'] = ((features_df.index.month % 12 + 3)//3).map(seasons)
    
    ohe_columns = ['day_name']
    features_df = pd.get_dummies(features_df, columns=ohe_columns, drop_first=True)
    return features_df 


def show_features_importances(model, features, n_splits, scoring, cluster_indx, target_col_name='n_trips'):
    """
    Plots feature importance for a given cluster

    model: Class (e.g. sklearn model/pipeline)
        Model 
    features: pd.DataFrame
        DataFrame with features 
    n_splits: int 
        Number of splits for Time Series Cross Validation 
    scoring: str
        Metric to minimize during cross validation
    cluster_indx: int
        Cluster index to show features importance for 
    target_col_name: str
        Target feature name
    Returns:
    -------
    None 
    """
    X_train = features.drop(columns=target_col_name)
    y_train = features[target_col_name]

    cv = cross_val_score(model, X_train, y_train,
                        cv=TimeSeriesSplit(n_splits),
                        scoring=scoring,
                        n_jobs=-1)
    
    cv_mae = round(cv.mean()*(-1),2)

    model.fit(X_train, y_train)
    coefs = pd.DataFrame(model[1].coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["Importance"] = coefs.coef.apply(np.abs)

    lasso_features = coefs["Importance"].sort_values(ascending=False)

    plt.figure(figsize=(15, 7))
    sns.barplot(lasso_features.index, lasso_features)
    plt.tight_layout()
    plt.xticks(rotation=65)
    plt.title(f'Feate Importances Cluster {cluster_indx} CV MAE: {cv_mae} Alpha: {model[1].alpha_}');


def get_lags(df, target_col_name, lag_start=1, lag_end=48, drop_target_col=True):
    """
    Computes lag features 

    df: pd.DataFrame
        Time Series Data
    target_col_name: str
        Target feature name
    lag_start: int
        Number of beginning lag 
    lag_end: int
        Number of the last lag 
    drop_target_col: bool
        Wether a target column should be dropped
    Returns:
    -------
    DataFrame with lag features 
    """
    features_df = df.copy()
    for i in range(lag_start, lag_end+1):
        features_df[f"lag_{i}"] = features_df[target_col_name].shift(i)
    features_df = features_df.dropna(axis='rows')
    if drop_target_col:
        features_df = features_df.drop(columns=target_col_name)
    return features_df


def get_harmonics(x_len, func, period, shift, factor):
    """
    Gets sin/cos harmonics 

    x_len: int 
        Time Series length 
    func: str
        Sin/Cos function
    period: int 
        Season length (e.g. 24, 48, 168...)
    shift: int 
        shift value (np.arrange())
    factor: int 
        Hyperparameter value 
    Returns:
    -------
    DataFrame
    """
    x = np.arange(1 + shift, x_len + shift + 1)
    if func == 'sin': f = np.sin
    else: f = np.cos
    return f(x * 2. * np.pi * float(factor)/float(period))


def get_harmonics_df(df, func='sin'):
    """
    Gets a DataFrame with harmonic features 

    df: pd.DataFrame 
        Time Series data
    func: str
        Sin/Cos function
    Returns:
    -------
    DataFrame
    """
    features_df = pd.DataFrame()
    periods = [6, 12, 24, 168]
    for period in periods:
        for shift in range (0, int(period/2), 2):
            for factor in range (1, int(period/2 + 1), 3):
                feature_name = '{}_{}_{}_{}'.format(func, int(period), shift, factor)
                features_df[feature_name] = get_harmonics(df.shape[0], func, period, shift, factor)
    features_df.index = df.index
    return features_df


def downsize_features_pca(df, seed):
    """
    Downsizes features using PCA 

    df: pd.DataFrame 
        Time Series data
    seed: int 
        Random Seed value
    Returns:
    -------
    DataFrame 
    """
    pca = PCA(random_state=seed)
    pca.fit(df)
    exp_var_array = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.argwhere(exp_var_array > 0.9)[0])
    
    pca = PCA(n_components=n_components, random_state=seed)
    res_df = pca.fit_transform(df)
    return res_df


def get_rolling_window_features(series, target_col_name, window_size=[12, 24], statistics=['avg'], drop_target_col=False):
    """
    Computes provided statistics using a rolling window

    series: pd.DataFrame or pd.Series 
        Time Series data
    target_col_name: str
        Target feature name
    window_size: list
        Size of the rolling window 
    statistics: list
        Statistcs to calcualte 
    drop_target_col: bool 
        If target column should be dropped 
    Returns:
    -------
    DataFrame
    """
    res_df = pd.DataFrame()
    res_df['n_trips'] = series['n_trips']
    for statistic in statistics:
        for size in window_size:
            if statistic == 'avg':
                res_df[f'rolling_{statistic}_{size}'] = series[target_col_name].rolling(size).mean()
            elif statistic == 'min':
                res_df[f'rolling_{statistic}_{size}'] = series[target_col_name].rolling(size).min()
            elif statistic == 'max':
                res_df[f'rolling_{statistic}_{size}'] = series[target_col_name].rolling(size).max()
            elif statistic == 'sum':
                res_df[f'rolling_{statistic}_{size}'] = series[target_col_name].rolling(size).sum()
            elif statistic == 'std':
                res_df[f'rolling_{statistic}_{size}'] = series[target_col_name].rolling(size).std()
    res_df = res_df.dropna()
    if drop_target_col:
        res_df = res_df.drop(columns=target_col_name)
    return res_df


def get_expanding_window_features(series, target_col_name, window_size=[12, 24], statistics=['avg'], drop_target_col=False):
    """
    Computes provided statistics using a expanding window

    series: pd.DataFrame or pd.Series 
        Time Series data
    target_col_name: str
        Target feature name
    window_size: list
        Size of the rolling window 
    statistics: list
        Statistics to calcualte 
    drop_target_col: bool 
        If target column should be dropped 
    Returns:
    -------
    DataFrame
    """
    res_df = pd.DataFrame()
    res_df['n_trips'] = series['n_trips']
    for statistic in statistics:
        for size in window_size:
            if statistic == 'avg':
                res_df[f'expanding_{statistic}_{size}'] = series[target_col_name].expanding(size).mean()
            elif statistic == 'min':
                res_df[f'expanding_{statistic}_{size}'] = series[target_col_name].expanding(size).min()
            elif statistic == 'max':
                res_df[f'expanding_{statistic}_{size}'] = series[target_col_name].expanding(size).max()
            elif statistic == 'sum':
                res_df[f'expanding_{statistic}_{size}'] = series[target_col_name].expanding(size).sum()
            elif statistic == 'std':
                res_df[f'expanding_{statistic}_{size}'] = series[target_col_name].expanding(size).std()
    res_df = res_df.dropna()
    if drop_target_col:
        res_df = res_df.drop(columns=target_col_name)
    return res_df