from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
import json

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

wikidata_all = sc.textFile("dumps/latest-all.json.bz2")


DISAMBIGUATION = 'Q4167410'
LIST = 'Q13406463'
INTERNAL_ITEM = 'Q17442446'
CATEGORY = 'Q4167836'

def get_entity_info(line):
    try:
        if DISAMBIGUATION in line or LIST in line or INTERNAL_ITEM in line or CATEGORY in line:
            return []
        row = json.loads(line[:-1])
        if 'type' in row and row['type'] == 'item':
            titles = []
            if 'sitelinks' in row:
                for k,v in row['sitelinks'].items():
                    site = v['site']
                    if site.endswith('wiki'):
                        title = v['title']
                        titles.append(Row(qid=row['id'], site=site, title=title))
            return titles
        else:
            return []
    except Exception as e:
        return []


articles = wikidata_all.flatMap(get_entity_info)

all_entities = spark.createDataFrame(articles)

all_entities.write.mode("overwrite").parquet("datasets/WikidataInterlanguageLinks.parquet")
