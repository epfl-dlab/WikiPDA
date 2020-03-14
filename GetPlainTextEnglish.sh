git clone https://github.com/attardi/wikiextractor #16186e290d9eb0eb3a3784c6c0635a9ed7e855c3
python ./wikiextractor/WikiExtractor.py -c -ns 0 --no_templates --filter_disambig_pages --discard_elements gallery,timeline,noinclude --json -b 128M --processes 48 -o datasets/enwiki/plain_text/ dumps/enwiki-20200220-pages-articles-multistream.xml.bz2
