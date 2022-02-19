import os 
import numpy as np
import pandas as pd 
import geopandas as gpd
import warnings 
from tqdm import tqdm

warnings.filterwarnings('ignore')


# Получение имен файлов за интересующий период 
def get_file_names(date_start, date_end):
    """
    Iterates over all files and returns only files for provided period 

    date_start: str ('YYYY-DD')
        Trips Start Date.
    date_end: str ('YYYY-DD')
        Trips End Date
    Returns
    -------
    List
        List with file names for specified period
    """
    files = os.listdir()
    
    result = []
    
    for file in files:
        file_date = file.split('.')[0].split('_')[-1]
        if file_date >= date_start and file_date <= date_end:
            result.append(file)
    return result


# Preprocessing для файла с геометрией 
def preprocess_polygon_data(f_name, file_path):
    """
    Preprocesses GeoDataFrame

    file_path: str
        File path with data (data location)
    f_name: str
        File name
    Returns
    -------
    GeoDataFrame
        Preprocessed GeoDataFrame
    """
    os.chdir(file_path)

    # Загружаем зоны такси с геометрией 
    polygon_data = gpd.read_file(f_name)
    col_to_use = polygon_data.columns.to_list()[-4:]
    polygon_data = polygon_data[col_to_use]

    # Изменяем тип location_id на int 
    polygon_data['location_id'] = polygon_data['location_id'].apply(pd.to_numeric)

    # Сортируем location_id
    polygon_data = polygon_data.sort_values(by='location_id').reset_index(drop=True)
    
    # Исправляем location_id
    polygon_data.loc[56, 'location_id'] = 57
    polygon_data.loc[103, 'location_id'] = 104
    polygon_data.loc[104, 'location_id'] = 105
    return polygon_data


# Фильтрация и отбор данных 
def filter_data(f_name, unique_districts):
    """
    Runs the main preprocessing for a Trips DataFrame 

    f_name: str
        File name
    unique_districts: list
        list with unique districts of New York city
    Returns
    -------
    DataFrame
        Preprocessed DataFrame
    """

    data = pd.read_csv(f_name, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    # Исключаем зоны для которых нет polygon_data
    data = data[data['PULocationID'].isin(unique_districts)]

    # Исключаем нулевое количество пассажиров (0 или NaN)
    data = data[data['passenger_count'] != 0]
    data = data[~data['passenger_count'].isnull()]

    # Исключаем нулевое расстояние 
    data = data[data['trip_distance'] != 0]

    # Исключаем нулевую длительность поездки а также поездки меньше 35c
    temp_series = (data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime'])
    res_indecies = temp_series[temp_series > pd.Timedelta("35 s")].index
    data = data.loc[res_indecies, :]
    
    # Удаляем дубликаты tpep_pickup_datetime
    data = data[~data[['tpep_pickup_datetime']].duplicated()]
    
    # Отбрасываем минуты и секунды во времени начала поездки (понадобится позже для агрегации)
    data = data.assign(tpep_pickup_datetime=pd.to_datetime(data['tpep_pickup_datetime'].dt.date) +
                        pd.to_timedelta(data['tpep_pickup_datetime'].dt.hour, unit='H'))
    
    # Число пасссажиров не должно превышать 6 человек
    data = data[~(data['passenger_count'] > 6)]
    
    # Отбираем правильные RatecodeID 
    unique_rate_codes = np.arange(1,7)
    rate_codes_to_drop = [code for code in data.RatecodeID.unique() if code not in unique_rate_codes]
    data = data[~(data['RatecodeID'].isin(rate_codes_to_drop))]
    
    # Даты не должны выходить за пределы текущей
    curr_year = int(f_name.split('.')[0].split('_')[-1].split('-')[0])
    curr_month = int(f_name.split('.')[0].split('_')[-1].split('-')[1])
    data = data.loc[(data.tpep_pickup_datetime.dt.year == curr_year) & (data.tpep_pickup_datetime.dt.month == curr_month)]
    return data


# Обработка пропусков в ряде (некоторые файлы имеют пропуски дат)
def fill_missing_dates(df, unique_districts):
    """
    Determines wether a data has missing dates. 
    If so, fills with missing date and random district  

    df: pd.DataFrame
        Preprocessed DataFrame
    unique_districts: list
        list with unique districts of New York city
    Returns
    -------
    DataFrame
        DataFrame with no missing dates
    """

    # Определяем уникальные даты текущего DF, берем начало и конец
    unique_dates = pd.to_datetime(df['tpep_pickup_datetime'].unique())
    dt_start = unique_dates[0]
    dt_end = unique_dates[-1]
    
    # Генерируем список всех дат с интревалом в час - это все даты которые должны быть
    all_dates = set(pd.date_range(dt_start, dt_end, freq = 'h'))
    df_dates = set(unique_dates) # множество дат в текущем DF
    missing_dates = all_dates.difference(df_dates) # определяем пропущенные даты 
    
    # Пропущенные даты заполняем случайными районами
    missing_dates_len = len(missing_dates)
    rand_districts = np.random.choice(list(unique_districts), missing_dates_len)
    
    # Добавляем пропущенные даты 
    if missing_dates:
        print('Missing Date Found: ', missing_dates)
        df = df.append(pd.DataFrame({
            'tpep_pickup_datetime': list(missing_dates), # т.к. set - unordered 
            'PULocationID': rand_districts,
            'n_trips': [0]*len(rand_districts)
        }))
        
        return df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    return df


# Заполянем отсутствующие пары час-регион (т.е. в каждом часе должны быть все районы города)
def fill_missing_districts_hours_pairs(df, unique_districts):
    """
    Fills mising hour-district pairs with n_trips = 0 value 

    df: pd.DataFrame
        Preprocessed DataFrame
    unique_districts: list
        list with unique districts of New York city
    Returns
    -------
    DataFrame
        DataFrame with no missing hour-district pairs
    """

    unique_dates = pd.to_datetime(df['tpep_pickup_datetime'].unique())
    for date in unique_dates:
        not_missing_districts = set(df[df['tpep_pickup_datetime'] == date]['PULocationID'])
        missing_districts = set(unique_districts).difference(not_missing_districts)
        
        if missing_districts:
            df = df.append(pd.DataFrame({
                'tpep_pickup_datetime': [date]*len(missing_districts),
                'PULocationID': list(missing_districts),
                'n_trips': [0]*len(missing_districts)
            })) 
    return df.sort_values('tpep_pickup_datetime').reset_index(drop=True)


# Провекрка корректности заполнения fill_missing_districts_hours_pairs() 
def check_missing_districts_hours_pairs(df, unique_districts_len=263):
    """
    df: pd.DataFrame
        Preprocessed DataFrame
    unique_districts_len: int
        Number of unique districts
    Returns
    -------
    Bool
        Wether shape is correct
    """
    assert df['tpep_pickup_datetime'].unique().shape[0]*unique_districts_len == df.shape[0]


# Получение среднего числа поездок в конкретный час за месяц 
def get_mean_trip_hourly(df):
    """
    df: pd.DataFrame
        Preprocessed DataFrame
    Returns
    -------
    DataFrame
        DataFrame with hourly mean n_trips for a certain month  
    """

    out_df = df.copy()
    
    # Извлекаем часы, так как по дате и времени сгруппировать не получитсся - все даты уникальные
    out_df['hour'] = df['tpep_pickup_datetime'].dt.time
    out_df = out_df.groupby(['hour', 'PULocationID'])['n_trips'].mean().reset_index(name='mean_n_trips')
    out_df['mean_n_trips'] = round(out_df['mean_n_trips'])
    
    # Добавляем дату в формате (YYYY-MM-HH:MM:SS) / совершаем обратную операцию
    dates = df['tpep_pickup_datetime'].map(lambda x: x.strftime('%Y-%m'))
    out_df['date'] = pd.to_datetime(dates + ' ' + out_df['hour'].astype('str'))
    out_df.drop(columns=['hour'], inplace=True)
    return out_df

# Основной Preprocessing Pipeline
def get_series_data(file_path, start_date, end_date, unique_districts, get_monthly_avg=True):
    """
    file_path: str
        File path with data (data location)
    start_date: str ('YYYY-DD')
        Start Date for Trips Data 
    end_date: str ('YYYY-DD')
        End Date for Trips Data
    unique_districts: list
        list with unique districts of New York city
    get_monthly_avg: bool
        if hourly mean n_trips for within a month is needed
    Returns 
    -------
    DataFrames
        One DataFrame with only n_trips and mean n_trips for within a month if get_monthly_avg=True
    """ 

    # Список необходимых файлов 
    os.chdir(file_path)
    files_to_process = get_file_names(date_start=start_date, date_end=end_date)
    
    # Результирующие DF's
    trips_df = pd.DataFrame() # число поездок на конкретную дату в конкретном районе города 
    mean_trips_df =  pd.DataFrame() # среднее число поездок в час в конкретном районе города за месяц
    
    # Основной Pipeline
    for file in tqdm(files_to_process):
        # 1. Фильтрация данных 
        data = filter_data(f_name=file, unique_districts=unique_districts)
        # 2. Аггрегация/Группировка
        data = data.groupby(['tpep_pickup_datetime', 'PULocationID']).size().reset_index(name='n_trips')
        # 3. Заполнение пропущенных дат 
        data = fill_missing_dates(df=data, unique_districts=unique_districts)
        # 4. Заполнение отсутствующей пары час-регион и проверка 
        data = fill_missing_districts_hours_pairs(df=data, unique_districts=unique_districts)
        check_missing_districts_hours_pairs(df=data)
        # 5. Добавляем обработанные данные за месяц в результирующий DF
        trips_df = trips_df.append(data)
        # Если нам нужны средние показатели в час по месяцам для каждого района города 
        if get_monthly_avg:
            data_monthly_avg = get_mean_trip_hourly(data)
            mean_trips_df = mean_trips_df.append(data_monthly_avg)
    
    # Переопределяем индекс
    trips_df = trips_df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    mean_trips_df = mean_trips_df.sort_values('date').reset_index(drop=True)
    return trips_df, mean_trips_df