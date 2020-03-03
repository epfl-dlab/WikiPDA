import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import urllib
import argparse


conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.jars.packages', 'com.databricks:spark-xml_2.11:0.8.0'),
                                   ('spark.driver.memory','64g'),
                                   ('spark.driver.maxResultSize', '32G'),
                                   ('spark.local.dir', '/scratch/tmp/'),
                                   ('spark.yarn.stagingDir', '/scratch/tmp/')      
                                  ])
# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext

parser = argparse.ArgumentParser()
parser.add_argument('--site', action="store")
args = parser.parse_args()

site = args.site  # 'en'

matched_ngrams = spark.read.parquet("datasets/{}/matched_ngrams.parquet".format(site))\
                .selectExpr("anchors as ngram", "qid", "occ")

ngrams_as_text = matched_ngrams.groupBy("ngram").agg(sum("occ").alias("as_text"))
commonness = spark.read.parquet("datasets/{}/anchors_stats.parquet".format(site))
ngram_as_link = commonness.groupBy("anchor").agg(sum("anchor_count").alias("as_link"))

ngrams_as_text.registerTempTable("ngrams_as_text")
ngram_as_link.registerTempTable("ngram_as_link")

query = """
SELECT anchor, as_link, CASE WHEN as_text is NULL THEN 0 ELSE as_text END as as_text
FROM ngram_as_link l
LEFT JOIN ngrams_as_text t
ON t.ngram=l.anchor
"""

anchors_beta = spark.sql(query).selectExpr("*", "as_link/(as_text+as_link) as beta")

anchors_beta.write.mode("overwrite").parquet("datasets/{}/anchors_beta.parquet".format(site))
