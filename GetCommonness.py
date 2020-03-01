from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import argparse

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','150g'),
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

anchors_info_qid = spark.read.parquet("datasets/{}/anchors_info_qid.parquet".format(site))

anchors_destinations = spark.createDataFrame(
    anchors_info_qid.rdd.map(lambda row: ((row.anchor, row.destination_qid), 1))
    .reduceByKey(lambda a, b: a + b)
    .map(lambda row: Row(anchor=row[0][0], destination=row[0][1], anchor_count=row[1])))

# Count how many time one anchor appears in wikipedia
anchors_counts = spark.createDataFrame(anchors_info_qid.rdd.map(lambda row: (row.anchor, 1))
                                            .reduceByKey(lambda a, b: a + b)
                                            .map(lambda row: Row(anchor=row[0], total=row[1])))

anchors_destinations.registerTempTable('anchors_destinations')
anchors_counts.registerTempTable('anchors_counts')

# Compute the ratio (anchor commonness)
anchors_stats_query = """
select ad.*, ac.total, ad.anchor_count / ac.total commonness
from anchors_destinations ad
join anchors_counts ac
on ac.anchor = ad.anchor
"""

anchors_stats = spark.sql(anchors_stats_query)
anchors_stats.write\
        .mode('overwrite')\
        .parquet("datasets/{}/anchors_stats.parquet".format(site))
