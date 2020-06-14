import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import urllib
import argparse
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.clustering import LDA

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','230g'),
                                   ('spark.driver.maxResultSize', '128G'),
                                   ('spark.local.dir', '/scratch/tmp/'),
                                   ('spark.yarn.stagingDir', '/scratch/tmp/'),
                                   ('spark.sql.warehouse.dir', '/scratch/tmp/')
                                  ])

# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# create the context
sc = spark.sparkContext


parser = argparse.ArgumentParser()
parser.add_argument('--k', action="store")
args = parser.parse_args()
k = int(args.k)

traning_set = spark.read.parquet("models/EnrichedLinks/limit500_traning_set.parquet")


lda = LDA(k=k, seed=42, maxIter=50)
model = lda.fit(traning_set)
model.write().overwrite().save("models/EnrichedLinks/limit500_LDA_model_{}_50iter.model".format(k))