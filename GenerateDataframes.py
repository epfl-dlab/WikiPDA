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
                                   ('spark.driver.memory','90g'),
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


WIKIPEDIA_DUMP = 'dumps/{}-20200220-pages-articles-multistream.xml.bz2'.format(site)
WIKIDATA_MAPPING = 'dumps/WikidataInterlanguageLinks.parquet'

wikipedia_all = spark.read.format('com.databricks.spark.xml') \
    .options(rowTag='page').load(WIKIPEDIA_DUMP) \
    .filter("ns = '0'") \
    .filter("revision.text._VALUE is not null") \
    .filter("length(revision.text._VALUE) > 0")

def normalise_title(title):
    """ Replace _ with space, remove anchor, capitalize """
    title = urllib.parse.unquote(title)
    title = title.strip()
    if len(title) > 0:
        title = title[0].upper() + title[1:]
    n_title = title.replace("_", " ")
    if '#' in n_title:
        n_title = n_title.split('#')[0]
    return n_title


def extract_article(row):
    """ Extract the content of the article"""
    redirect = row.redirect._title if row.redirect is not None else None
    return Row(
        id=row.id,
        title=normalise_title(row.title),
        redirect=redirect,
        text=row.revision.text._VALUE
    )

wikipedia = spark.createDataFrame(wikipedia_all.rdd.map(extract_article).filter(lambda r: r is not None))

wikipedia.write.mode("overwrite").parquet("datasets/{}/wikipedia.parquet".format(site))
wikipedia = spark.read.parquet("datasets/{}/wikipedia.parquet".format(site))

# get the redirects: from --> to
redirects = spark.createDataFrame(
    wikipedia.filter("redirect is not null")
        .rdd.map(lambda r: Row(from_title=r.title,
                           to_title=normalise_title(r.redirect)))
).distinct()

articles = wikipedia.filter("redirect is null")

normalise_title_udf = udf(normalise_title)
wikidata = spark.read.parquet(WIKIDATA_MAPPING)\
                     .select("qid", "site", normalise_title_udf("title").alias("title"))\
                     .where(col('site') == site)

articles_qid = articles.alias("a").join(wikidata, articles.title==wikidata.title).select("a.*", "qid")


redirects_qid = redirects.alias("r")\
        .join(wikidata, redirects.to_title==wikidata.title)\
        .selectExpr("r.*", "qid as destination_qid")


import re
links_regex = re.compile(r"\[\[(?P<link>[^\n\|\]\[\<\>\{\}]{0,256})(?:\|(?P<anchor>[^\[]*?))?\]\]")

import urllib
def normalise_anchor(anchor):
    anchor = urllib.parse.unquote(anchor)
    n_anchor = anchor.strip().replace("_", " ")
    return n_anchor.lower()

def get_links(page):
    links = []
    for m in links_regex.findall(page.text):
        link = normalise_title(m[0])
        anchor = m[1] if len(m) > 1 and len(m[1]) > 0 else link
        if len(link) > 0:
            links.append(Row(page_id=page.id,
                             qid=page.qid,
                             title=page.title,
                             link=link,
                             anchor=normalise_anchor(anchor)))
    return links


links = spark.createDataFrame(articles_qid.rdd.flatMap(get_links))

links.registerTempTable("links")
redirects.registerTempTable("redirects")

# Replace the anchor with original destination title if not specified
anchors_query = """
SELECT page_id, qid, title, anchor, link, CASE WHEN ISNULL(r.to_title) THEN l.link ELSE r.to_title END AS destination
FROM links l
LEFT JOIN redirects r
ON l.link = r.from_title
"""
anchors_info = spark.sql(anchors_query)
anchors_info_qid = anchors_info.alias("ai")\
                    .join(wikidata.alias("w"), anchors_info.destination==wikidata.title)\
                    .selectExpr("ai.*", "w.qid as destination_qid")

anchors_info_qid.write\
    .mode("overwrite")\
    .parquet("datasets/{}/anchors_info_qid.parquet".format(site))

anchors_info_qid = spark.read.parquet("datasets/{}/anchors_info_qid.parquet".format(site))
# get the number of incoming links for a page
incoming_links_count = anchors_info_qid.groupBy("destination_qid").agg(count("qid").alias("entities_count"))

# Assign a row index for the concept
matrix_positions = spark.createDataFrame(anchors_info_qid.select("qid").distinct()
                                         .rdd.map(lambda r: r.qid)
                                         .zipWithIndex().map(lambda r: Row(qid=r[0], matrix_index=r[1])))


N = matrix_positions.count()

import math
# Compute weight of each page -log(in_links/N)
pos_weight_rdd = matrix_positions.join(incoming_links_count,
                                       incoming_links_count.destination_qid == matrix_positions.qid) \
    .rdd.map(lambda r: Row(qid=r.qid, matrix_index=r.matrix_index, weight=-math.log(float(r.entities_count) / N)))
pos_weight = spark.createDataFrame(pos_weight_rdd)

pos_weight.registerTempTable("pos_weight")
anchors_info_qid.registerTempTable("anchors_info_qid")

# Create the matrix with weights
query = """
SELECT l.qid as source_qid, l.destination_qid, row.matrix_index row, col.matrix_index col, row.weight
FROM anchors_info_qid l
JOIN pos_weight row
JOIN pos_weight col
ON l.qid = row.qid
AND l.destination_qid = col.qid
"""

matrix_entries = spark.sql(query).distinct()
matrix_entries.write\
    .mode('overwrite').parquet("datasets/{}/matrix_entries.parquet".format(site))
matrix_positions.write\
    .mode('overwrite').parquet("datasets/{}/matrix_positions.parquet".format(site))

references_regex = re.compile(r"<ref[^>]*>[^<]+<\/ref>")
def get_plain_text_without_links(row):
    """ Replace the links with a dot to interrupt the sentence and get the plain text """
    wikicode = row.text
    wikicode_without_links = re.sub(links_regex, '.', wikicode)
    wikicode_without_links = re.sub(references_regex, '.', wikicode_without_links)
    try:
        text = mwparserfromhell.parse(wikicode_without_links).strip_code()
    except:
        text = wikicode_without_links
    return Row(id=row.id, title=normalise_title(row.title), text=text.lower(), qid=row.qid)


plain_text_without_links = spark.createDataFrame(articles_qid.rdd.map(get_plain_text_without_links))

plain_text_without_links.write\
    .mode('overwrite')\
    .parquet("datasets/{}/plain_text_without_links.parquet".format(site))



