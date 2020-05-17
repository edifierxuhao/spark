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
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import split
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
data = 'df_for_ml.json'
df_for_ml = spark.read.json(data)
df_for_ml.persist()

train, test = df_for_ml.randomSplit([0.8,0.2], seed = 42)


assembler1 = VectorAssembler(
        inputCols=['Action_ph', 'Session_ph', 'Nextsong_ph',
        'Downgrade_ph', 'Upgrade_ph', 'ThumbDown_ph', 'ThumbUp_ph', 'Home_ph',
        'Adv_ph','Addtolist_ph','Set_ph','Addfriend_ph','Error_ph', 'Help_ph',
        'Action_toSession', 'Nextsong_toAct', 'Downgrade_toAct', 'Upgrade_toAct',
        'ThumbDown_toAct', 'ThumbUp_toAct','Home_toAct','Adv_toAct',
        'Addtolist_toAct', 'Set_toAct','Addfriend_toAct','Error_toAct',
        'Help_toAct', 'Action_trend','Nextsong_trend','Nextsong_betweenHome',
        ],
        outputCol='NumFeatures')

scaler = Normalizer(inputCol='NumFeatures',
                    outputCol='ScaledNumFeatures',
                    p = 1.0)

assembler2 = VectorAssembler(inputCols = ['gender_num','level_num','ScaledNumFeatures'],
                             outputCol = 'features')

rf = RandomForestClassifier(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages = [assembler1, scaler, assembler2, rf])


paramGrid = ParamGridBuilder()\
        .addGrid(scaler.p,[1.0,2.0])\
        .addGrid(rf.maxDepth,[5, 10]) \
        .addGrid(rf.numTrees, [20, 50]) \
        .addGrid(rf.minInstancesPerNode, [1, 10]) \
        .addGrid(rf.subsamplingRate, [0.7, 1.0]) \
        .build()

crossval = CrossValidator(estimator = pipeline,
                          estimatorParamMaps = paramGrid,
                          evaluator = MulticlassClassificationEvaluator(),
                          numFolds = 3)




cvModel = crossval.fit(train)
best_model = cvModel.bestModel
result = best_model.transform(test)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                                                labelCol="label")

f1_score = evaluator.evaluate(result, {evaluator.metricName: "f1"})
print(result.filter(result.label == result.prediction).count()/result.count())
print(f1_score)
