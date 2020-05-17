# import libraries
from pyspark.sql import SparkSession

from pyspark.sql.functions import udf, desc, asc, sum, max, min, avg, countDistinct,\
            row_number, col, expr, round,first, count
from pyspark.sql.types import StringType, IntegerType, LongType
from pyspark.sql import Window

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.ml.feature import Normalizer, StandardScaler, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import split
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# build spark session
spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "50g") \
    .config('spark.executor.memory', '70G')\
    .config('spark.driver.maxResultSize', '10G')\
    .config("spark.memory.offHeap.enabled",True)\
    .config("spark.memory.offHeap.size","16g")\
    .appName("Project") \
    .getOrCreate()

# load data and clean data
data = 'mini_sparkify_event_data.json'
df = spark.read.json(data)
df_valid = df.dropna(how = 'any',subset = ['userId','sessionId'])\
             .filter(col('userId') != '' )
df_valid.persist()

# Clean and wrangling dataframe
# window functions
windowval = Window.partitionBy('userId')\
                  .orderBy('ts')\
                  .rangeBetween(Window.unboundedPreceding,0)
windowval_desc = Window.partitionBy('userId')\
                  .orderBy(desc('ts'))\
                  .rangeBetween(Window.unboundedPreceding,0)
windowval_desc_nobound = Window.partitionBy('userId')\
                               .orderBy(desc('ts'))
flag_cancel_confirm_event = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0,IntegerType())
gender_to_number = udf(lambda x: 1 if x== 'F' else 0,IntegerType())
level_to_num = udf(lambda x: 0 if x== 'free' else 1,IntegerType())
tem_df = df_valid.groupby('userId').agg(count('page').alias('actions'))
# transform the gender column to 0 and 1
df_valid = df_valid.withColumn('gender_num', gender_to_number('gender'))\
                   .drop('gender')
# transform the level column to 0 and 1
df_valid = df_valid.withColumn('level_num', level_to_num('level'))\
                   .drop('level')
# built two new columns, to save the first and the last timestamp
df_valid = df_valid.withColumn('first_ts', first('ts').over(windowval))\
                   .withColumn('last_ts', first('ts').over(windowval_desc))
# built a column to save the active hours between first_ts and last_ts
df_valid = df_valid.withColumn('active_time',col('last_ts').cast('long') - col('first_ts').cast('long'))\
                   .withColumn('active_hour',round((col('active_time')/3600000),2))\
                   .drop('active_time')
# built a columns to save the time passed from first_ts to this record
df_valid = df_valid.withColumn('passed_time',col('ts').cast('long') - col('first_ts').cast('long'))\
                   .withColumn('passed_hour',round((col('passed_time')/3600000),2))\
                   .drop('passed_time')
# built a columns to save the time passed from this record to the last_ts
df_valid = df_valid.withColumn('time_till_last', col('last_ts').cast('long')- col('ts').cast('long'))\
                   .withColumn('hour_till_last',round((col('time_till_last')/3600000),2))\
                   .drop('time_till_last')
# add a column to save the time order of all records for each user
df_valid = df_valid.withColumn('latest_level', row_number().over(windowval_desc_nobound))
# add Churn feature, if the user churned marked all records with 1, else 0
df_valid = df_valid.withColumn('cancle_confirmed', flag_cancel_confirm_event('page'))\
                   .withColumn('Churn', sum('cancle_confirmed').over(windowval))\
                   .drop('cancle_confirmed')
# add a column to save the all action counts for each user
df_valid = df_valid.join(tem_df, on = ['userId'], how = 'left' )


# extract the necessary features to a new dataframe

# userId + categorical features
new_df = df_valid.where(df_valid.latest_level == 1)\
                 .select('userId','Churn','gender_num','level_num')


# Numerical pre hour features
# Action_ph_df
Action_ph_df = df_valid.groupby('userId')\
                       .agg(count('artist').alias('action'),max('active_hour').alias('active_hour'))\
                       .withColumn('Action_ph', round(col('action')/col('active_hour'),8))\
                       .drop('action','active_hour')
new_df = new_df.join(Action_ph_df, on=['userId'], how='left')
# Session_ph_df
Session_ph_df = df_valid.groupby('userId')\
                        .agg(countDistinct('sessionId').alias('sessionId'),max('active_hour').alias('active_hour'))\
                        .withColumn('Session_ph', round(col('sessionId')/col('active_hour'),8))\
                        .drop('sessionId','active_hour')
new_df = new_df.join(Session_ph_df, on=['userId'], how='left')
# all item in page column
name_dict = {'NextSong':'Nextsong_ph',
             'Downgrade':'Downgrade_ph',
             'Upgrade':'Upgrade_ph',
             'Thumbs Down':'ThumbDown_ph',
             'Thumbs Up':'ThumbUp_ph',
             'Home':'Home_ph',
             'Roll Advert':'Adv_ph',
             'Add to Playlist':'Addtolist_ph',
             'Settings':'Set_ph',
             'Add Friend':'Addfriend_ph',
             'Error':'Error_ph',
             'Help':'Help_ph'}
for item in name_dict.keys():
    temp_df = df_valid.where(df_valid.page == item)\
                              .groupby('userId')\
                              .agg(count('page').alias('page'),max('active_hour').alias('active_hour'))\
                              .withColumn(name_dict[item], round(col('page')/col('active_hour'),8))\
                              .drop('page','active_hour')
    new_df = new_df.join(temp_df, on=['userId'], how='left')


# Ratio of Action Features
# Action_toSession
Action_toSession_df = df_valid.groupby('userId','sessionId')\
            .agg({'page':'count'})\
            .groupby('userId')\
            .agg(round(avg('count(page)'),8).alias('Action_toSession'))
new_df = new_df.join(Action_toSession_df, on=['userId'], how='left')
# other page items to action ratio
name_dict2 = {'NextSong':'Nextsong_toAct',
             'Downgrade':'Downgrade_toAct',
             'Upgrade':'Upgrade_toAct',
             'Thumbs Down':'ThumbDown_toAct',
             'Thumbs Up':'ThumbUp_toAct',
             'Home':'Home_toAct',
             'Roll Advert':'Adv_toAct',
             'Add to Playlist':'Addtolist_toAct',
             'Settings':'Set_toAct',
             'Add Friend':'Addfriend_toAct',
             'Error':'Error_toAct',
             'Help':'Help_toAct'}
for item in name_dict2.keys():
    temp_df = df_valid\
        .where(df_valid.page == item)\
        .groupby('userId')\
        .agg(count('page').alias('page'),max('actions').alias('actions'))\
        .withColumn(name_dict2[item], round(col('page')/col('actions'),8))\
        .drop('page','actions')
    new_df = new_df.join(temp_df, on=['userId'], how='left')

# Time trend Features


# Action_trend
Action_last_14day = df_valid\
        .where(df_valid.hour_till_last <= 336)\
        .groupby(col('userId'))\
        .agg(count('page').alias('last14'))
Action_first_14day = df_valid\
        .where(df_valid.passed_hour <= 336)\
        .groupby(col('userId'))\
        .agg(count('page').alias('first14'))
Action_trend_df = Action_last_14day\
        .join(Action_first_14day, on = ['userId'], how = 'inner')\
        .withColumn('Action_trend',round(col('last14')/(col('first14')+0.01),8))\
        .drop('last14','first14')
# Nextsong_trend
Nextsong_last_14day = df_valid\
        .where((df_valid.hour_till_last <= 336) & (df_valid.page == 'NextSong'))\
        .groupby(col('userId'))\
        .agg(count('page').alias('last14'))
Nextsong_first_14day = df_valid\
        .where((df_valid.passed_hour <= 336) & (df_valid.page == 'NextSong'))\
        .groupby(col('userId'))\
        .agg(count('page').alias('first14'))
Nextsong_trend_df = Nextsong_last_14day\
        .join(Nextsong_first_14day, on = ['userId'], how = 'inner')\
        .withColumn('Nextsong_trend',round(col('last14')/(col('first14')+0.01),8))\
        .drop('last14','first14')
new_df = new_df\
        .join(Action_trend_df, on=['userId'], how='left')\
        .join(Nextsong_trend_df, on=['userId'], how='left')


# Behavioral Features
# Nextsong_betweenHome show the average nextsong palyed between two home pages for each user
function = udf(lambda ishome : int(ishome == 'Home'), IntegerType())
Nextsong_betweenHome_df = df_valid\
        .filter((df_valid.page == 'NextSong') | (df_valid.page == 'Home'))\
        .select('userId', 'page', 'ts')\
        .withColumn('homevisit', function(col('page')))\
        .withColumn('period', sum('homevisit')\
        .over(windowval_desc))\
        .filter((col('page') == 'NextSong'))\
        .groupBy('userId', 'period')\
        .agg({'period':'count'})\
        .groupBy('userId')\
        .agg(round(avg('count(period)'),8).alias('Nextsong_betweenHome'))
new_df = new_df.join(Nextsong_betweenHome_df, on=['userId'], how='left')

# Deal with null values
df_for_ml = new_df\
        .drop('userId')\
        .withColumnRenamed('Churn','label')\
        .na.fill(0)

df_for_ml.write.format('json').save('df_for_ml.json')
