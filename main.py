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

def pre_process(tdf):
    tdf = tdf.na.replace('', None)
    tdf = tdf.na.drop()       
    tdf = tdf.withColumn('feature1', regexp_replace('feature1',r'@\w+',''))
    tdf = tdf.withColumn('feature1', regexp_replace('feature1','[^A-Za-z0-9\s]',''))
    tdf = tdf.withColumn('feature1', regexp_replace('feature1',r'http\S+',''))
    return tdf

df = pre_process(tweetdf3)


def p(b,id):
    tokenizer = Tokenizer(inputCol="feature1", outputCol="words")
    pipeline = Pipeline(stages=[tokenizer])
    
    pipelineFit = pipeline.fit(b)
    ddf = pipelineFit.transform(b)
    ddf.show()
    
query = df.writeStream.foreachBatch(p).start()
query.awaitTermination()
