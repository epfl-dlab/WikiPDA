for site in arwiki dewiki elwiki enwiki eswiki fawiki fiwiki frwiki hewiki idwiki itwiki kowiki nlwiki plwiki ptwiki rowiki ruwiki sqwiki srwiki trwiki ukwiki viwiki warwiki zhwiki; do
  wget "https://dumps.wikimedia.org/${site}/20200220/$site-20200220-pages-articles-multistream.xml.bz2" -P dumps
done 

wget https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
