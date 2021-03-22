from wikipda.article import Preprocessor, Bagger, fetch_article_data
from wikipda.model import WikiPDAModel, ORESClassifier

# Scrape articles from Wikipedia using revision IDs
titles, revisions, wikitexts = fetch_article_data([985175359], 'en')

# Also possible to process only wikitexts!
p = Preprocessor('en')
articles = p.load(wikitexts, revisions, titles, enrich=True)

# Create bag-of-links representations
b = Bagger()
bols = b.bag(articles)

# Produce embeddings
model = WikiPDAModel(k=40)
topics_distribution = model.get_distribution(bols)

# predict the categories of the texts
# NOTE: currently only supports topic embeddings with k=300
classifier = ORESClassifier()
text_categories = classifier.predict_category(topics_distribution)

print(topics_distribution)
print(text_categories)
