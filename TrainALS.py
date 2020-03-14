from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import urllib
import mwparserfromhell
import argparse

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.jars.packages', 'com.databricks:spark-xml_2.11:0.8.0'),
                                   ('spark.driver.memory','230g'),
                                   ('spark.driver.maxResultSize', '32G'),
                                   ('spark.local.dir', '/scratch/tmp/'),
                                   ('spark.yarn.stagingDir', '/scratch/tmp/'),
                                   ('spark.sql.warehouse.dir', '/scratch/tmp/')
                                  ])

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# create the context
sc = spark.sparkContext

parser = argparse.ArgumentParser()
parser.add_argument('--site', action="store")
args = parser.parse_args()
site = args.site

matrix_entries = spark.read.parquet("datasets/{}/matrix_entries.parquet".format(site))
anchors_info_qid = spark.read.parquet("datasets/{}/anchors_info_qid.parquet".format(site))

beta = spark.read.parquet("datasets/{}/anchors_beta.parquet".format(site))
beta_filter = beta.where("as_link >= 10 and beta >= 0.065").select("anchor")

anchors_info_qid_filtered = anchors_info_qid.join(beta_filter, "anchor")
candidates = anchors_info_qid_filtered.filter("anchor not rlike '^[0-9]+$'").where("LENGTH(anchor)>0")\
    .groupBy("anchor").agg(collect_set("destination_qid").alias("candidates"))
ambiguous = candidates.where("size(candidates)>1")
ambiguous_links = anchors_info_qid_filtered.join(ambiguous, "anchor")\
        .select("qid", "anchor", "destination_qid", "candidates").distinct()

ambiguous_links_testing_set = ambiguous_links.sample(0.05, seed=123)

ambiguous_links_testing_set\
        .write.mode("overwrite")\
        .parquet("datasets/{}/ambiguous_links_testing_set.parquet".format(site))


matrix_entries.registerTempTable("matrix_entries")
ambiguous_links_testing_set.registerTempTable("ambiguous_links_testing_set")


query = """
SELECT met.*
FROM matrix_entries met
LEFT JOIN ambiguous_links_testing_set alt
ON met.source_qid=alt.qid
AND met.destination_qid=alt.destination_qid
WHERE (alt.qid is NULL
AND alt.destination_qid is NULL)
"""

matrix_entries_no_amb_testing = spark.sql(query)

ratings = matrix_entries_no_amb_testing.rdd.map(lambda r: Rating(r.row, r.col, r.weight)).cache()

model = ALS.trainImplicit(ratings, 300, 10, lambda_=0.1, alpha=0.01)

model.save(sc, "datasets/{}/ALSModel.model".format(site))
