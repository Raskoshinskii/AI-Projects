import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

import pandas as pd
import geopandas as gpd
import numpy as np
import warnings 
import os

from tqdm import tqdm
warnings.filterwarnings('ignore')


def spark_shape(self):
    return (self.count(), len(self.columns))
pyspark.sql.dataframe.DataFrame.shape = spark_shape


def get_values(df_col, col_name):
    """
    Returns colums values of a Spark DataFrame

    df_col: list 
        List with Spark Row Objects 
    col_name: str
        Column name 
    Returns
    -------
    List
        Column values list 
    """
    values = [x[col_name] for x in df_col]
    return values


def copy_spark_df(spark_df, schema):
    """
    Copies Spark DataFrame

    spark_df: spark.DataFrame
        Spark DataFrame
    schema: pyspark.sql.types.StructType
        Spark Schema 
    Returns:
    -------
    DataFrame
        Copy of Spark DataFrame
    """
    pandas_df = spark_df.toPandas()
    copy_df = spark.createDataFrame(pandas_df,schema=schema)
    del pandas_df
    return copy_df


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
def process_polygon_data(f_name, file_path):
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
    polygon_data = gpd.read_file(f_name)
    col_to_use = polygon_data.columns.to_list()[-4:]
    polygon_data = polygon_data[col_to_use]
    polygon_data['location_id'] = polygon_data['location_id'].apply(pd.to_numeric)
    polygon_data = polygon_data.sort_values(by='location_id').reset_index(drop=True)
    polygon_data
    polygon_data.loc[56, 'location_id'] = 57
    polygon_data.loc[103, 'location_id'] = 104
    polygon_data.loc[104, 'location_id'] = 105
    return polygon_data


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
        Preprocessed Spark DataFrame
    """
    data = spark.read.csv(f_name, header=True, inferSchema=True)

    data = data\
        .withColumn('tpep_pickup_datetime', F.col('tpep_pickup_datetime').cast(TimestampType()))\
        .withColumn('tpep_dropoff_datetime', F.col('tpep_dropoff_datetime').cast(TimestampType()))
    
    data = data.filter(F.col('PULocationID').isin(unique_districts))
    
    data = data.filter(F.col('passenger_count') != 0.0)
    data = data.filter(F.col('passenger_count').isNotNull())

    data = data.filter(F.col('trip_distance') != 0.0)

    data = data.withColumn("date_diff_seconds", F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long"))
    data = data.filter(F.col('date_diff_seconds') > 35.0)
    
    data = data.dropDuplicates(["tpep_pickup_datetime"])

    data = data.withColumn('tpep_pickup_datetime', F.date_trunc('hour', F.col('tpep_pickup_datetime')))

    data = data.filter(~(F.col('passenger_count') > 6))

    unique_rate_codes = np.arange(1,7)
    rate_codes_to_drop = [code['RatecodeID'] for code in data.select('RatecodeID').distinct().collect() if code['RatecodeID'] not in unique_rate_codes]
    data = data.filter(~(F.col('RatecodeID').isin(rate_codes_to_drop)))
    
    curr_year = int(f_name.split('.')[0].split('_')[-1].split('-')[0])
    curr_month = int(f_name.split('.')[0].split('_')[-1].split('-')[1])

    data = data.filter(
        (year(F.col('tpep_pickup_datetime')) == curr_year) & \
        (month(F.col('tpep_pickup_datetime')) == curr_month)
    )
    return data


def fill_missing_dates(df, unique_districts):
    """
    Determines wether a data has missing dates. 
    If so, fills with missing date and random district  

    df: spark.DataFrame
        Preprocessed DataFrame
    unique_districts: list
        list with unique districts of New York city
    Returns
    -------
    DataFrame
        Spark DataFrame with no missing dates
    """
    unique_dates = df.select('tpep_pickup_datetime').distinct().orderBy('tpep_pickup_datetime').collect()
    unique_dates = list(map(lambda x: x['tpep_pickup_datetime'], unique_dates))

    dt_start = unique_dates[0]
    dt_end = unique_dates[-1]

    all_dates = set(pd.date_range(dt_start, dt_end, freq = 'h').to_pydatetime())
    df_dates = set(unique_dates)
    missing_dates = all_dates.difference(df_dates)
    missing_dates_len = len(missing_dates)
    rand_districts = np.random.choice(unique_districts, missing_dates_len)
    
    if missing_dates:
        print('Missing Date Found: ', missing_dates)
        pandas_df = pd.DataFrame({
            'tpep_pickup_datetime': list(missing_dates),
            'PULocationID': rand_districts,
            'n_trips': [0]*len(rand_districts)
        })
        spark_df = spark.createDataFrame(pandas_df)
        df = df.union(spark_df) 
    return df.orderBy('tpep_pickup_datetime')

def fill_missing_districts_hours_pairs(df, unique_districts):
    """
    Fills mising hour-district pairs with n_trips = 0 value 

    df: spark.DataFrame
        Preprocessed DataFrame
    unique_districts: list
        list with unique districts of New York city
    Returns
    -------
    DataFrame
        Spark DataFrame with no missing hour-district pairs
    """
    unique_dates = df.select('tpep_pickup_datetime').distinct().collect()
    unique_dates = list(map(lambda x: x['tpep_pickup_datetime'], unique_dates))
    for date in unique_dates:
        current_date_districts = df.filter(F.col('tpep_pickup_datetime') == date).select('PULocationID').collect()
        date_districts = set(get_values(current_date_districts, 'PULocationID'))
        missing_districts = set(unique_districts).difference(date_districts)
        if missing_districts:
            pandas_df = pd.DataFrame({
                'tpep_pickup_datetime': [date]*len(missing_districts),
                'PULocationID': list(missing_districts),
                'n_trips': [0]*len(missing_districts)
            })
            spark_df = spark.createDataFrame(pandas_df)
            df = df.union(spark_df)
    return df.orderBy('tpep_pickup_datetime')


def check_missing_districts_hours_pairs(df, unique_districts_len=263):
    """
    Cheks if fill_missing_districts_hours_pairs() worked corectly

    df: spark.DataFrame
        Preprocessed Spark DataFrame
    unique_districts_len: int
        Number of unique districts
    Returns
    -------
    Bool
        Wether shape is correct
    """
    assert df.select('tpep_pickup_datetime').distinct().shape()[0]*unique_districts_len == df.shape()[0]


def get_mean_trip_hourly(df):
    """
    df: spark.DataFrame
        Preprocessed Spark DataFrame
    Returns
    -------
    DataFrame
        Spark DataFrame with hourly mean n_trips for a certain month  
    """
    df = df.withColumn('datetime', date_format(F.col('tpep_pickup_datetime'), 'yyyy-MM HH:mm:ss'))
    df = df.drop('tpep_pickup_datetime')
    df = df\
        .groupBy('datetime', 'PULocationID')\
        .agg(
            F.round(F.mean(F.col('n_trips'))).alias('mean_n_trips')
        )
    df = df.withColumn('datetime', F.col('datetime').cast(T.TimestampType()))
    return df

def create_schema(columns, types, is_nullabel):
    """
    columns: list 
        List with column names 
    types: list 
        List with certain column type names
    is_nullabel: list 
        List with True/False values
    Returns:
    -------
    Schema
        Spark DataFrame Schema 
    """
    col_types = []
    for i, col in enumerate(columns):
        col_types.append(StructField(col, types[i], is_nullabel[i]))
    return StructType(col_types)


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
    os.chdir(file_path)
    files_to_process = get_file_names(date_start=start_date, date_end=end_date)
    # Resulting DF's (using PySpark you need to predefine the data schema)
    # We already know in advance what will be in the DF's data, so we will pass the schema using create_schema()
    trips_schema = create_schema(columns=['tpep_pickup_datetime', 'PULocationID', 'n_trips'],
                                types = [TimestampType(), IntegerType(), LongType()],
                                is_nullabel = [True, True, False])
    
    mean_trips_schema = create_schema(columns=['datetime', 'PULocationID', 'mean_n_trips'],
                                    types = [TimestampType(), IntegerType(), DoubleType()],
                                    is_nullabel = [True, True, True])
    
    trips_df = spark.createDataFrame([], schema=trips_schema) 
    mean_trips_df = spark.createDataFrame([], schema=mean_trips_schema) 
    for file_name in tqdm(files_to_process):
        data = filter_data(f_name='taxi_data/'+file_name, unique_districts=unique_districts)
        data = data.groupBy('tpep_pickup_datetime', 'PULocationID').agg(F.count(F.col('passenger_count')).alias('n_trips'))
        data = fill_missing_dates(df=data, unique_districts=unique_districts)
        data = data.toPandas()
        data = fill_missing_districts_hours_pairs(df=data, unique_districts=unique_districts)
        data = spark.createDataFrame(data)
        check_missing_districts_hours_pairs(df=data)
        trips_df = trips_df.union(data)
        if get_monthly_avg:
            data_monthly_avg = get_mean_trip_hourly(data)
            mean_trips_df = mean_trips_df.union(data_monthly_avg)
    return trips_df, mean_trips_df