for site in arwiki dewiki elwiki enwiki eswiki fawiki fiwiki frwiki hewiki idwiki itwiki kowiki nlwiki plwiki ptwiki rowiki ruwiki sqwiki srwiki trwiki ukwiki viwiki warwiki zhwiki; do
  spark-submit --conf spark.driver.memory=90g GetNGrams.py --site $site
done 
