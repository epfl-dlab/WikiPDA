import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
from pyspark.ml.clustering import LocalLDAModel
from pyspark.ml.feature import CountVectorizerModel
import numpy
import argparse

conf = pyspark.SparkConf().setMaster("local[48]").setAll([
                                   ('spark.driver.memory','250g'),
                                   ('spark.driver.maxResultSize', '32G'),
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

sites = [ 
         "eswiki", "itwiki", "dewiki", "frwiki", "cswiki", 
         "cawiki", "svwiki", "arwiki", "elwiki", "fiwiki", 
         "hewiki", "idwiki", "kowiki", "nlwiki", "plwiki", 
         "ptwiki", "rowiki", "ruwiki", "sqwiki", "srwiki", 
         "trwiki", "ukwiki", "viwiki", "warwiki", "zhwiki", 
         "fawiki", "jawiki", "nowiki", "huwiki", "dawiki", 
         "bgwiki", "slwiki", "hrwiki", "hiwiki", "euwiki", 
         "mswiki", "thwiki", "eowiki", "hywiki", "kawiki", 
         "etwiki", "ltwiki", "shwiki", "azwiki"]

all_links_enriched_rdd = sc.emptyRDD()
for s in sites:
    links = spark.read.parquet("datasets/{}/enriched_all_links.parquet".format(s))
    all_links_enriched_rdd = all_links_enriched_rdd.union(links.selectExpr("'{}' as site".format(s), "*").rdd)
    
all_links_enriched = spark.createDataFrame(all_links_enriched_rdd)


model = LocalLDAModel.load("models/EnrichedLinks/limit500_LDA_model_{}.model".format(k))
transformer = CountVectorizerModel.load("models/EnrichedLinks/limit500_transformer.model")

result = transformer.transform(all_links_enriched).cache()
topics_distribution = model.transform(result)

topics_distribution.selectExpr("site", "qid", "topicDistribution", "SIZE(links) as total_links")\
    .write.mode("overwrite").parquet("models/Embeddings/EnrichedALL/all_articles_{}.parquet".format(k))
