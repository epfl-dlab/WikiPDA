# WikiPDA
This is the library for the cross-lingual topic modelling technique 
known as [WikiPDA](https://arxiv.org/abs/2009.11207). It allows you 
to do the following:
- Fetch wikipedia articles by their revision IDs
- Process these articles in accordance with the paper
- Produce the topic embeddings for these
- Produce text category predictions for the embeddings

## User Manual
### Installing
Since the package isn't published yet you can just run the following command to install
the package: `pip install git+https://github.com/epfl-dlab/wikipda-lib.git`

In order to use this API you first need to download all the resources
required. These are available in the following 
[drive folder](https://drive.google.com/drive/u/2/folders/1V4Gyx870NWLbPy9d3H1FPDRpzyJ5eVzu).
Here is what you need to do:
1. Create a folder in your home directory named `wikipda_data`
2. Add the resources from the drive document in a a subfolder called `lang`.
    1. E.g., for English you would have to download the `en.tar.gz` tarball from the
drive folder. 
    2. Then extract it to `wikipda_data/lang/en`.
3. You then need to put the following files/folders from drive in `wikipda_data`:
    - `classifiers`
    - `LDA_models`
    - `topic_dict.pkl`
    
If you would prefer another placement of resources you can configure this in 
`wikipda/settings.py`.

## Example usage
```python
from wikipda.article import Preprocessor, fetch_article_data
from wikipda.model import LDAModel, TextClassifier

# Scrape rticles from Wikipedia using revision IDs
titles, revisions, wikitexts = fetch_article_data([985175359], 'en')

# Also possible to process only wikitexts!
p = Preprocessor('en')
articles = p.load(wikitexts, revisions, titles, enrich=True)

# Produce embeddings
model = LDAModel(k=300)
embeddings = model.get_embeddings(articles)

# predict the categories of the texts
# NOTE: currently only supports topic embeddings with k=300
classifier = TextClassifier()
text_categories = classifier.predict_category(embeddings)
```
