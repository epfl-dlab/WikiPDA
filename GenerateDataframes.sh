for site in arwiki dewiki elwiki enwiki eswiki fawiki fiwiki frwiki hewiki idwiki itwiki kowiki nlwiki plwiki ptwiki rowiki ruwiki sqwiki srwiki trwiki ukwiki viwiki warwiki zhwiki; do
  spark-submit --packages com.databricks:spark-xml_2.11:0.8.0 --conf spark.driver.memory=90g GenerateDataframes.py --site $site
done 
