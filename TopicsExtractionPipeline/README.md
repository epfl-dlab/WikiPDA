# Pre-processing pipeline

These scripts constitute the processing pipeline to extract the enriched list of links. These scripts require Python 3.x and PySpark used in local mode. The code can be adapted to work on a distributed YARN cluster by customizing the creation of the Spark context (first lines in each Python file). Please refer to the Apache Spark [documentation](https://spark.apache.org/docs/latest/index.html) for more information.

The input is a set of XML dumps in the format released monthly by the Wikimedia Foundation, and the output is a set a [Parquet](https://spark.apache.org/docs/latest/sql-data-sources-parquet.html) files (one per language).

To make sure the process is easy to debug, the code is modular and organised in multiple scripts that must be executed in sequence.

Note: replace `<LIST OF LANGUAGES>` with the Wikipedia edition name separated by space: i.e., `enwiki itwiki dewiki frwiki`

1. Download the Wikitext of the different languages. For the article `Crosslingual Topic Modeling withWikiPDA` we used the dump released the 20th of February 2020 (20200220). Consider to use a more recent dump if the files are not available anymore. The files are downloaded in the folder `dumps`.

```
for site in <LIST OF LANGUAGES>; do
  wget "https://dumps.wikimedia.org/${site}/20200220/$site-20200220-pages-articles-multistream.xml.bz2" -P dumps
done
```

2. Parse the XML in the folder `dumps` and generate the dataframes. The script requires the library [spark-xml](https://github.com/databricks/spark-xml). The output is written in the folder `datasets/<language_name>`.

```
for site in <LIST OF LANGUAGES>; do
  spark-submit --packages com.databricks:spark-xml_2.11:0.8.0 GenerateDataframes.py --site $site
done
```

List of files generated:

* `wikipedia.parquet`: parsed XML in convenient Parquet format.
* `anchors_info_qid.parquet`: List of the original links in the page with anchor text and redirections resolved.
* `matrix_entries.parquet`: Adjacency matrix of the links in a page. It contains an entry for all non-zero transitions. The entries are rescaled in the spirit of inverse document frequency: $-log(d_j/N)$
* `matrix_positions.parquet`: Simple mapping $QID$ <=> matrix position
* `plain_text_without_links.parquet`: Text with the links replaced by `.` (dot).

3. Get the commonness values for all the anchors in the dataset.

```
for site in <LIST OF LANGUAGES>; do
  spark-submit GetCommonness.py --site $site
done
```

This script generates the file:

* `anchors_stats.parquet`: Anchor text, occurrences count and commonness.

4. Generate the n-grams with `n = 1..4`, match the strings with the existing anchors and save them. The matching starts from the largest `n` value and when a match is found, the substring is removed.

```
for site in <LIST OF LANGUAGES>; do
  spark-submit GetNGrams.py --site $site
done
```

This script generates the file:

* `matched_ngrams.parquet`: Dataframe with the matched anchors for each page with their relative frequency.

5. Compute the beta parameter: frequency of the anchor as link divided the total number of occurrences. This is used to remove string used rarely as link (i.e., stopwords).

```
for site in <LIST OF LANGUAGES>; do
  spark-submit GetBeta.py --site $site
done
```

This script generates the file:

* `anchors_beta.parquet`: Dataframe with the matched anchors for each page with their relative frequency.

6. [Factorise](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html) the links matrix with [ALS](https://dl.acm.org/doi/10.1109/MC.2009.263). By default the matrix is factorised in 300 latent dimensions.

```
for site in <LIST OF LANGUAGES>; do
  spark-submit TrainALS.py --site $site
done
```

This script generates the file:

* `ALSModel.model`: Resulting Spark ALS model. To map the matrix index to the relative article $QID$ use the file `matrix_positions.parquet`.

7. Generate the dataset with enriched links. This script merges the original links, the links for non-ambiguos anchors, and it assigns the link for ambiguos anchors by using the relateness metric. Anchors with more than 10 candidates are discarded.

```
for site in <LIST OF LANGUAGES>; do
  spark-submit GetEnrichedDataset.py --site $site
done
```

This script generates the file:

* `enriched_all_links.parquet`: The dataframe contains the enriched bag of links.

<hr>

# Merging and training LDA

Once the previous scripts generated the enriched set of links for all the languages, we can produce the language-agnostic bag-of-link representation and train the LDA model.

This can be done with the following 2 scripts:

1. Merge the documents (append), filter the links with less than 500 occurrences across all languages, filter out the documents with less than ten links, and generate the vector representation. 

Note: edit the variable `sites` of the script to specify the languages to include. 

```
spark-submit MergeAndVectorizeLinks.py
```

This script generates in the folder `models/EnrichedLinks` the files:

* `limit500_transformer.model`: The transformer that converts a bag of $QIDs$ in vector representation. It contains the dictionary to use as reverse-index. Please refer to the [official documentation](https://spark.apache.org/docs/latest/ml-features#countvectorizer).
* `limit500_traning_set.parquet`: Vectoral representation of each document.

2. Finally you can train the LDA model for a specific value of K with:

```
spark-submit EnrichedLinks_TrainLDA.py --k <VALUE OF K>
```
The resulting model is save in `models/EnrichedLinks/` with the name `limit500_LDA_model_K_50iter.model`, where `K` is the selected value.

<hr>

# Generate embedding

The script `GetEmbeddingEnriched.py` provides an example of how to generate the topics distribution (or embedding vector) for all the documents in the dataset.

Example:
```
spark-submit GetEmbeddingEnriched.py --k <VALUE OF K>
```

The resulting model is save in `models/Embeddings/EnrichedALL/` with the name `all_articles_K.parquet`, where `K` is the selected value.

This script can be used as reference to generate the topics vector for any bag-of-links.