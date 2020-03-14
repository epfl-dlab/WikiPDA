from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json
import urllib
import re
import argparse


conf = pyspark.SparkConf().setMaster("local[*]").setAll([
                                   ('spark.driver.memory','230g'),
                                   ('spark.driver.maxResultSize', '32G'),
                                   ('spark.local.dir', '/scratch/tmp/'),
                                   ('spark.yarn.stagingDir', '/scratch/tmp/'),
                                   ('spark.sql.warehouse.dir', '/scratch/tmp/')
                                  ])

# conf = pyspark.SparkConf().setMaster("yarn")


# create the session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
# create the context
sc = spark.sparkContext

parser = argparse.ArgumentParser()
parser.add_argument('--site', action="store")
args = parser.parse_args()

site = args.site  # 'en'


anchors_info_qid = spark.read.parquet("datasets/{}/anchors_info_qid.parquet".format(site))
plain_text_without_links = spark.read.parquet("datasets/{}/plain_text_without_links.parquet".format(site))

anchors_stats = spark.read.parquet("datasets/{}/anchors_stats.parquet".format(site)).where("total >= 10").select("anchor").distinct()
anchors_info_qid_filtered = anchors_info_qid.join(anchors_stats, "anchor")

valid_anchors = anchors_info_qid_filtered.where("LENGTH(anchor)>0")\
                .filter("anchor not rlike '^[0-9]+$' and anchor not rlike '^[0-9]+[/-][0-9]+$'")\
                .groupBy("anchor").agg(count("*").alias("total"))\
                .where("total>1")

anchors = set(valid_anchors.select("anchor").rdd.map(lambda r: r.anchor).collect())

def get_chunks(row):
    return [Row(qid=row.qid, chunk=blocks.strip()) for blocks in re.split('[\n\.,;:()!"]', row.text) 
            if len(blocks.strip())>0]

chunks = plain_text_without_links.rdd.flatMap(get_chunks)

def get_ngrams(txt, n):
    ngrams = []
    words = txt.split()
    if len(words)>=n:
        ngram_list = [words[i:i+n] for i in range(len(words)-n+1)]
        for e in ngram_list:
            ngrams.append(" ".join(e))
    return ngrams

def get_valid_ngrams(row):
    text = row.chunk
    found_anchors = []
    for n in range(4, 0, -1):
        ngrams = get_ngrams(text, n)
        for ng in ngrams:
            if ng in anchors:
                found_anchors.append(ng)
                text.replace(ng, " @ ")
    return [Row(qid=row.qid, anchor=a) for a in found_anchors]

matched_ngrams = spark.createDataFrame(chunks.flatMap(get_valid_ngrams))\
        .groupBy("anchor", "qid").agg(count("*").alias("occ"))

matched_ngrams.write\
        .mode('overwrite')\
        .parquet("datasets/{}/matched_ngrams.parquet".format(site))

