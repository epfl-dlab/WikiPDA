# from wikipda.article import Preprocessor, Bagger, fetch_article_data
# from wikipda.model import WikiPDAModel, ORESClassifier
#
# # Scrape articles from Wikipedia using revision IDs
# titles, revisions, wikitexts = fetch_article_data([985175359], 'en')
#
# # Also possible to process only wikitexts!
# p = Preprocessor('en')
# articles = p.load(wikitexts, revisions, titles, enrich=True)
#
# # Create bag-of-links representations
# b = Bagger()
# bols = b.bag(articles)
#
# # Produce embeddings
# model = WikiPDAModel(k=300)
# topics_distribution = model.get_distribution(bols)
#
# # predict the categories of the texts
# # NOTE: currently only supports topic embeddings with k=300
# classifier = ORESClassifier()
# text_categories = classifier.predict_category(topics_distribution)
#
# print(topics_distribution)
# print(text_categories)


from wikipda.article import Preprocessor, Bagger, fetch_article_data
from wikipda.model import WikiPDAModel, ORESClassifier

# Scrape articles from Wikipedia using revision IDs
# titles, revisions, wikitexts = fetch_article_data([89694880], 'ja')
#
# a={'text': '{{Musiikkialbumi\n| levy = Kamenno sarce\n| tyyppi = studio\n| kansi = Kamenno sarce.jpg\n| leveys = 260\n| artisti = [[Lijana]]\n| julkaistu = 19. marraskuuta 1998<ref name="itunes">[https://itunes.apple.com/us/album/kamenno-sarce/563970970 iTunes: LiANA – Kamenno sarce {{en}}]</ref>\n| tuottaja = \n| nauhoitettu = \n| genre = [[pop-folk]]\n| minuutit = \n| sekunnit = \n| levy-yhtiö = Orfei Music\n| artistin = [[Lijana]]\n| vuosit = 1998\n| edellinen = \n| vuosie = \n| seuraava = [[Platinena žena]] \n| vuosis = 2000\n|}}\n\'\'\'Kamenno sarce\'\'\' ({{K-bg|Каменно сърце}}) [[bulgaria]]laisen pop-folk -laulaja [[Lijana]]n esikoisalbumi, joka julkaistiin 19. marraskuuta 1998.<ref name="itunes"/>\n\n== Kappaleet ==\n# Kamenno sarce\n# Posledna rana\n# Tše si tarsja\n# Angel beli\n# Ne me laži\n# Ne me moli\n# Tancuvaj s men\n# Vino ili ženi\n# Nalei mi vino\n# Tažna jesen\n# Pisna me\n\n== Lähteet ==\n{{viitteet}}\n\n{{Tynkä/Albumi}}\n[[Luokka:Vuoden 1998 esikoisalbumit]]\n[[Luokka:Bulgarialaiset popalbumit]]', 'page_id': 1430373, 'revision_id': 17054164, 'title': 'Kamenno sarce'}
#
# # Also possible to process only wikitexts!
# p = Preprocessor('fi')
# # articles = p.load(wikitexts, revisions, titles, enrich=True)
# articles = p.load([a['text']], [a['revision_id']], [a['title']], enrich=True)
#
#
# # Create bag-of-links representations
# b = Bagger()
# bols = b.bag(articles)
#
# # Produce embeddings
# model = WikiPDAModel(k=300)
# topics_distribution = model.get_distribution(bols)
#
# # predict the categories of the texts
# # NOTE: currently only supports topic embeddings with k=300
# classifier = ORESClassifier()
# text_categories = classifier.predict_category(topics_distribution)
#
# print(topics_distribution)
# print(text_categories)

titles, revisions, wikitexts = fetch_article_data([89600287], 'ja')

p = Preprocessor('ja')
articles = p.load(wikitexts, revisions, titles, enrich=True)
# articles = p.load([a['text']], [a['revision_id']], [a['title']], enrich=True)


# Create bag-of-links representations
b = Bagger()
bols = b.bag(articles)

# Produce embeddings
model = WikiPDAModel(k=300)
topics_distribution = model.get_distribution(bols)

# predict the categories of the texts
# NOTE: currently only supports topic embeddings with k=300
classifier = ORESClassifier()
text_categories = classifier.predict_category(topics_distribution)

print(topics_distribution)
print(text_categories)
