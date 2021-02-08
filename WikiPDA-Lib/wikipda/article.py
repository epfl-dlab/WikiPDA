"""
Module contains the code related to fetching and pre-processing given Wikipedia aritcle revisions.
In the end, using the ArticleGenerator object, you will have the revision id, language, QID links
and bag-of-links representation of the links for each article represented as an Article object.
"""

import pickle
import re
import os
from typing import List, Tuple, Dict
from wikipda.settings import TOPIC_DICT_PATH, RESOURCE_PATH
import numpy as np

from py_mini_racer import py_mini_racer
import requests


def fetch_article_data(revision_ids: List[int], language: str) \
        -> Tuple[List[str], List[str], List[str]]:
    """
    Takes a list of revision IDs and a given language, then retrieves the wikitext and title
    for that article. In the end it returns the title, revision ID and wikitext for all the
    revision IDs given.

    :param revision_ids: List of revision IDs to process.
    :param language: Language edition of Wikipedia to query.
    :return: Titles, revision IDs and wikitext representations of articles.
    """

    # Divide revision ids into subsets of 50.
    # MediaWiki limits the revisions for revision IDs to a maximum of 50 at a time.
    revision_ids = [str(revid) for revid in revision_ids]
    id_subsets = [revision_ids[i:i + 50] for i in range(0, len(revision_ids), 50)]
    id_subsets = ['|'.join(ids) for ids in id_subsets]

    all_data = []
    for ids in id_subsets:
        params = {
            'action': 'query',
            'prop': 'revisions',  # more details at https://www.mediawiki.org/wiki/API:Revisions
            'rvprop': 'content|ids',  # content=wikitext, ids=revision id
            'format': 'json',
            'revids': ids,
        }
        response = requests.get(f'https://{language}.wikipedia.org/w/api.php', params=params)
        response = response.json()['query']['pages']

        # for each entry...
        data = [(response[page_id]['title'],  # ...extract title...
                 revision['revid'],  # ...and revision ID
                 revision['*'])  # ...and wikitext
                for page_id in response.keys() for revision in response[page_id]['revisions']]
        all_data += data

    titles, revisions, raw_texts = map(tuple, zip(*all_data))
    return titles, revisions, raw_texts


class Article:
    """
    Class represents a processed and ready-to-go wikipedia article. Can be fed into the next
    step of the WikiPDA pipeline.
    """

    def __init__(self, revision_id, links, bol, language):
        self.revision_id = revision_id
        self.language = language
        self.links = links
        self.bol = bol


class Preprocessor:
    """
    Object for processing articles on Wikipedia using the WikiPDA preprocessing pipeline. Each
    instance is specific to a language edition of Wikipedia, since the different language editions
    use different resources.
    """

    def __init__(self, language: str):
        """
        Constructor which loads a lot of resources for the processing of articles. It takes a couple
        of seconds to load everything.

        :param language: Code for the language edition of wikipedia that you wish to process
        resources for. E.g: en (english), it (italian) or sv (swedish)
        """

        # Set the language for the articles
        self.language = language

        # Used for creating the bag-of-links representation later
        with open(TOPIC_DICT_PATH, 'rb') as f:
            self.topic_dictionary = pickle.load(f)

        # Used when mapping links in the Wikitext to their underlying QIDs
        with open(RESOURCE_PATH + language + '/title_qid_mappings.pickle', 'rb') as f:
            self.title_qid_mapping = pickle.load(f)

        # Get our javascript context started
        self.js_ctx = py_mini_racer.MiniRacer()

        # Load wtf_wikipedia (wikitext parsing library)
        library = open(os.path.join(os.path.dirname(__file__), 'wtf_wikipedia.js'), 'rb').read()
        self.js_ctx.eval(library)

        # Used when densifying articles with non-existing links
        with open(RESOURCE_PATH + language + '/qid_mappings.pickle', 'rb') as f:
            self.anchors = pickle.load(f)

        # Matrix factorization used for disambiguating ambiguous anchors
        # NOTE: these are accessed from disk using mmap
        self.users = np.load(RESOURCE_PATH + language + '/users.npy', mmap_mode='r')
        self.products = np.load(RESOURCE_PATH + language + '/products.npy', mmap_mode='r')

        # This contains the matrix indices in the factorization for articles
        # I.e., mapping from QID -> Index in matrix factorization
        with open(RESOURCE_PATH + language + '/matrix_positions.pickle', 'rb') as f:
            self.matrix_positions = pickle.load(f)

    @staticmethod
    def extract_links(raw_texts: List[str], title_qid_mapping: Dict,
                      js_ctx: py_mini_racer.MiniRacer) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Takes a given wikitext, extracts the links and then parses that text.

        :param raw_texts: The wikitext representation of Wikipedia articles.
        :param title_qid_mapping: Mapping between the titles and QIDs of Wikipedia articles.
        :param js_ctx: context that allows for running javascript. Needs to have executed the
        wtf_wikipedia library beforehand.
        :return: The QIDs and texts without wikicode.
        """

        links_regex = re.compile(
            r"\[\[(?P<link>[^\n\|\]\[\<\>\{\}]{0,256})(?:\|(?P<anchor>[^\[]*?))?\]\]")

        filtered_texts = []
        links = []
        for text in raw_texts:
            # Extract links
            links.append([match[0] for match in links_regex.findall(text)])

            # Remove links from plain text
            filtered = re.sub(links_regex, '.', text)
            filtered_texts.append(filtered.lower())

        # Parse and retrieve plaintext of articles without wikicode
        clean_texts = js_ctx.eval(f'var texts = {str(list(filtered_texts))}\n' +
                                  'var parsed = texts.map((text) => wtf(text))\n' +
                                  'parsed.map((parse) => parse.text())')

        # The links are currently represented as the titles of the articles
        # Iterate over them and retrieve their QIDs instead
        qid_links = []
        for link_set in links:
            qid_links_article = []
            for title in link_set:
                if title is not None:
                    qid = Preprocessor.get_qid_from_titles([title], title_qid_mapping)[0]
                    if qid is not None:
                        qid_links_article.append(qid)
            qid_links.append(qid_links_article)

        return qid_links, clean_texts

    @staticmethod
    def get_qid_from_titles(titles: List[str], title_qid_mapping: Dict) -> List[str]:
        """
        Retrieve the QID for a given wikipedia article title.

        :param titles: The titles to retrieve QIDs for.
        :param title_qid_mapping: The mapping to use when retrieving the QIDs.
        :return: The retrieved QIDs.
        """
        qids = []
        for title in titles:
            # Capitalize the first letter to match format in mapping
            capitalized = title[0].capitalize() + title[1:]
            v = title_qid_mapping.get(capitalized)
            qids.append(v)

        return qids

    @staticmethod
    def get_chunks(document: str) -> List[str]:
        """
        Chunks given document using punctuation and other tokens used to divide text.

        :param document: Document to chunk
        :return: The chunks of the document
        """
        return [blocks for blocks in re.split(r'[\n\.,;:()!"]', document)
                if len(blocks.strip()) > 0]

    @staticmethod
    def get_ngrams(txt: str, n: int) -> List[str]:
        """
        Takes given text and computes the n-grams of words for that text.

        :param txt: Text to compute n-grams for
        :param n: Amount of words to associate
        :return: Computed n-grams for document
        """
        ngrams = []
        words = txt.split()
        if len(words) >= n:

            # Divide the n-grams into separate list
            ngram_list = [words[i:i + n] for i in range(len(words) - n + 1)]

            # Re-join them as individual strings
            for e in ngram_list:
                ngrams.append(" ".join(e))
        return ngrams

    @staticmethod
    def get_links(document: str, anchors: Dict, users: np.array, products: np.array,
                  matrix_indices: Dict, wikidata_id: str, existing_links: List[str]) -> List[str]:
        """
        Produce wikipedia knowledge base concept IDs from plain text by using the given anchors.

        :param document: Document to anchor concept IDs from
        :param anchors: Mapping between the anchor texts and concept IDs
        :param users: The vector embeddings for the source articles in a matrix factorization model
        :param products: The vector embeddings for the candidate articles in a matrix
        factorization model
        :param matrix_indices: Mappings between qids and matrix indices for matrix factorization
        :param wikidata_id: Mappings between qids and matrix indices for matrix factorization
        :param existing_links: The links that were already present in the document explicitly
        :return: The retrieved concept IDs
        """

        # Iterate and process the chunks of the document
        chunks = Preprocessor.get_chunks(document)
        found_anchors = []
        mi = matrix_indices.get(wikidata_id)

        # Get embedding if already present
        if mi is not None:
            u = users[mi]

        # Otherwise compute embedding by projecting the links onto V
        # We only project it onto the columns of V which will not be 0 since
        # our link vector will be sparse
        else:

            # Compute adjacency vector
            N = products.shape[0]
            a = np.zeros(N)
            for link in existing_links:
                index = matrix_indices.get(link)
                if index is not None:
                    # Increment adjacency count
                    a[index] += 1

            # Create sparse matrix using needed rows
            sparse_V = np.zeros(products.shape)
            cis = np.nonzero(a)
            sparse_V[cis] = products[cis]

            # Compute embedding using product
            u = a @ sparse_V

        for chunk in chunks:

            # Create links by computing n-grams from n=4...1 and anchoring on the biggest
            # possible n-grams
            for n in range(4, 0, -1):

                # Process n-grams
                ngrams = Preprocessor.get_ngrams(chunk, n)
                for ng in ngrams:
                    qids = anchors.get(ng)

                    # Skip if anchor non-existent
                    if qids is None:
                        continue

                    # Disambiguate phrase
                    # NOTE: skips disambiguation of source article isn't in the matrix factorization
                    elif len(qids) > 1:

                        # Find the candidate links
                        candidates = [(c, matrix_indices.get(c))
                                      for c in qids if matrix_indices.get(c) is not None]

                        cs = [c for _, c in candidates]
                        candidate_products = products[cs]
                        scores = u @ candidate_products.T

                        # Select best candidate
                        best_candidate = candidates[np.argmax(scores)][0]
                        found_anchors.append(best_candidate)
                        chunk.replace(ng, " @@ ")

                    # Append found non-ambiguous anchor
                    elif qids is not None:
                        found_anchors.append(qids[0])
                        chunk.replace(ng, " @@ ")

        return found_anchors

    def load(self, wikitexts: List[str], revisions: List[str] = None,
             titles: List[str] = None, enrich: bool = True) -> List[Article]:
        """
        Produces Article objects for all the given wikitexts by running them through
        the full WikiPDA pipeline.

        :param wikitexts: The wikitext for all the articles to process.
        :param revisions: (Optional) The revision IDs for all the articles.
        :param titles: (Optional) The titles of all the articles.
        :param enrich: Whether to perform link densification or not.
        :return: The Article objects representing the final product of the processing pipeline.
        """

        # Get the qid of the articles (if available)
        if titles is not None:
            qids = Preprocessor.get_qid_from_titles(titles, self.title_qid_mapping)
        else:
            qids = [None] * len(wikitexts)

        # Extract existing links
        qid_links, clean_texts = Preprocessor.extract_links(
            wikitexts, self.title_qid_mapping, self.js_ctx)

        articles = []
        for i, clean_text in enumerate(clean_texts):

            # Densify if indicated
            if enrich:
                qid_links[i].extend(Preprocessor.get_links(clean_text,
                                                           self.anchors,
                                                           self.users,
                                                           self.products,
                                                           self.matrix_positions,
                                                           qids[i],
                                                           qid_links[i]))

            # Bag links
            bol = self.topic_dictionary.doc2bow(qid_links[i])

            # Article is ready
            revision = revisions[i] if revisions is not None else 'NA'
            articles.append(Article(revision, qid_links[i], bol, self.language))

        return articles
