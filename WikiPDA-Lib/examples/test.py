from wikipda.article import Preprocessor, fetch_article_data
from wikipda.model import LDAModel, TextClassifier

# Scrape articles from Wikipedia using revision IDs
titles, revisions, wikitexts = fetch_article_data([985175359], 'en')

# Also possible to process only wikitexts!
p = Preprocessor('en')
articles = p.load(wikitexts, revisions, titles, enrich=True)
# Produce embeddings
model = LDAModel(k=40)
embeddings = model.get_embeddings(articles)

# predict the categories of the texts
# NOTE: currently only supports topic embeddings with k=300
classifier = TextClassifier()
text_categories = classifier.predict_category(embeddings)

print(embeddings)
print(text_categories)