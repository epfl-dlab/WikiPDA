from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import argparse

conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','230g'),
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
parser.add_argument('--site', action="store")
args = parser.parse_args()

site = args.site  # 'en'


matched_ngrams = spark.read.parquet("datasets/{}/matched_ngrams.parquet".format(site))\
    .selectExpr("anchor", "qid", "occ")
anchors_info_qid = spark.read.parquet("datasets/{}/anchors_info_qid.parquet".format(site))
candidates = anchors_info_qid.filter("anchor not rlike '^[0-9]+$'").where("LENGTH(anchor)>0")\
    .groupBy("anchor").agg(collect_set("destination_qid").alias("candidates"))
matched_ngrams_candidates=matched_ngrams.join(candidates, "anchor")

not_ambiguous = matched_ngrams_candidates.where("SIZE(candidates)=1")
ambiguous = matched_ngrams_candidates.where("SIZE(candidates)>1 and SIZE(candidates)<=10")


# ADD REGULAR LINKS

valid_links_noself = anchors_info_qid.select("qid", "destination_qid")
self_links = anchors_info_qid.selectExpr("qid", "qid as destination_qid").distinct()
valid_links = valid_links_noself.union(self_links)

# ADD NON AMBIGUOUS

na_links = spark.createDataFrame(not_ambiguous.rdd.
            flatMap(lambda r: [Row(qid=r.qid, destination_qid=r.candidates[0]) for i in range(0, r.occ)]))

valid_links = valid_links.union(na_links)


#ADD AMBIGUOUS
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
model = MatrixFactorizationModel.load(sc, "datasets/{}/ALSModel.model".format(site))
matrix_positions = spark.read.parquet("datasets/{}/matrix_positions.parquet".format(site))

ambiguous_with_id = ambiguous.withColumn("instance_id", monotonically_increasing_id())

expanded = ambiguous_with_id.select("instance_id", "anchor", "qid", explode("candidates").alias("cand"))
expanded.registerTempTable("expanded")
matrix_positions.registerTempTable("matrix_positions")

query = """
SELECT e.*, row.matrix_index row, col.matrix_index col
FROM expanded e
JOIN matrix_positions row
JOIN matrix_positions col
ON e.qid = row.qid
AND e.cand = col.qid
"""

ambiguous_with_index = spark.sql(query)


data = ambiguous_with_index.rdd.map(lambda p: (p.row, p.col))
predictions_rdd = model.predictAll(data).map(lambda r: (Row(row=r[0], col=r[1], relateness=r[2])))
predictions = spark.createDataFrame(predictions_rdd)


ambiguous_with_index.registerTempTable("ambiguous_with_index")
predictions.registerTempTable("predictions")

query = """
SELECT a.*, p.relateness
FROM ambiguous_with_index a
JOIN predictions p
ON a.row=p.row
AND a.col=p.col
"""

ambiguous_with_relateness = spark.sql(query)

candidates_scores_rdd = ambiguous_with_relateness.rdd\
        .map(lambda r: (r.instance_id, [Row(candidate=r.cand, relateness=r.relateness)]))\
        .reduceByKey(lambda a,b: a+b)\
        .map(lambda r: Row(instance_id=r[0], best_match=sorted(r[1], key=lambda r: -r.relateness)[0].candidate))

candidates_scores = spark.createDataFrame(candidates_scores_rdd)

ambiguous_with_scores = ambiguous_with_id.join(candidates_scores, "instance_id").drop("candidates")
a_links = spark.createDataFrame(ambiguous_with_scores.rdd.
            flatMap(lambda r: [Row(qid=r.qid, destination_qid=r.best_match) for i in range(0, r.occ)]))

valid_links = valid_links.union(a_links).groupBy("qid").agg(collect_list("destination_qid").alias("links"))
valid_links.write.mode("overwrite").parquet("datasets/{}/enriched_all_links.parquet".format(site))




