import pyspark
import pyspark.sql
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer


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

sites = ["jawiki", "cswiki", "cawiki", "svwiki", "arwiki", "dewiki", 
             "elwiki", "enwiki", "eswiki", "fawiki", "fiwiki", "frwiki", 
             "hewiki", "idwiki", "itwiki", "kowiki", "nlwiki", "plwiki", 
             "ptwiki", "rowiki", "ruwiki", "sqwiki", "srwiki", "trwiki", 
             "ukwiki", "viwiki", "warwiki", "zhwiki"]

all_links_enriched_rdd = sc.emptyRDD()
for s in sites:
    links = spark.read.parquet("datasets/{}/enriched_all_links.parquet".format(s))
    all_links_enriched_rdd = all_links_enriched_rdd.union(links.selectExpr("'{}' as site".format(s), "*").rdd)
    
all_links_enriched = spark.createDataFrame(all_links_enriched_rdd)
all_links = all_links_enriched.select("site", explode("links").alias("link")).cache()
all_links_count = all_links.groupBy("site", "link").agg(count("*").alias("link_count")).cache()

# Drop links used less than 500 times across all 28 languages
all_links_500 = all_links_count.where("link_count > 500").selectExpr("link").distinct().cache()

# Add self loop
training_self_loop = spark.createDataFrame(all_links_enriched
                                           .rdd.map(lambda r: Row(qid=r.qid, site=r.site, links=r.links+[r.qid])))

training_all_links_rdd = training_self_loop.rdd.flatMap(
                    lambda r: [Row(site=r.site, qid=r.qid, link=l) for l in r.links])

training_all_links = spark.createDataFrame(training_all_links_rdd).cache()

# 32636147 in total
traning_filtered = training_all_links.join(all_links_500, "link")\
    .groupBy("qid", "site").agg(collect_list("link").alias("links")).cache()

# 20479100 in total
traning_filtered_10 = traning_filtered.where("SIZE(links)>=10")

# 437624
N = traning_filtered_10.select("site", explode("links").alias("link")).select("link").distinct().count()

# Get the bag of links
wordsVector = CountVectorizer(inputCol="links", outputCol="features", vocabSize=N)
transformer = wordsVector.fit(traning_filtered_10)
result = transformer.transform(traning_filtered_10).cache()

transformer.write().overwrite().save("models/EnrichedLinks/limit500_transformer.model")
result.selectExpr("SIZE(links) links_count", "features")\
    .write.mode("overwrite").parquet("models/EnrichedLinks/limit500_traning_set.parquet")