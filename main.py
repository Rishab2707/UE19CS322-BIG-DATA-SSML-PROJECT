from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import split,col
from pyspark.sql.types import *
import json

spark = SparkSession \
    .builder \
    .appName("Tweet analysis") \
    .getOrCreate()

text_schema = StructType([StructField('text', StringType(), True)])
json_schema = MapType(StringType(), StringType())

tweet_df= spark \
    .readStream \
    .format("socket")\
    .option("host", "localhost") \
    .option("port",6100) \
    .load()
tweet_df.printSchema()
tweet_df1= tweet_df.selectExpr("CAST(value AS STRING)")

tweetdf2=tweet_df1.withColumn('json',from_json(col('value'),json_schema)).select(explode(col('json')))
tweetdf3=tweetdf2.withColumn('value',from_json(col('value'),json_schema)).withColumn('feature0',col('value.feature0')).withColumn('feature1',col('value.feature1')).drop('value')
query = tweetdf3\
    .writeStream \
    .format("console") \
    .option('truncate',False)\
    .start()\
    

query.awaitTermination()

