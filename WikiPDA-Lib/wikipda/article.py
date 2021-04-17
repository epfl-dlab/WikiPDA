"""
Module contains the code related to fetching and pre-processing given Wikipedia aritcle revisions.
In the end, using the ArticleGenerator object, you will have the revision id, language, QID links
and bag-of-links representation of the links for each article represented as an Article object.
"""

import pickle
import sqlite3
import re
from typing import List, Tuple, Dict
from wikipda.settings import DATA_DIR, RESOURCE_PATH
import numpy as np
from collections import OrderedDict
import mwparserfromhell as mw
import logging

import requests

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


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
        response = response.json()['query']

        # Prompt user with information regarding which revisions were unavailable
        if 'badrevids' in response:
            bad_revids = response['badrevids'].keys()
            if len(bad_revids) > 0:
                e = f'The following revisions could not be retrieved from the MediaWiki API: \n' \
                    f'{", ".join(bad_revids)}\n' \
                    f'This is likely due to the given revision having been deleted from Wikipedia.\n' \
                    f'These revisions have been skipped.\n'
                logging.warning(e)

        # for each entry...
        response = response['pages']
        data = []
        for page_id in response.keys():
            for revision in response[page_id]['revisions']:
                title = response[page_id]['title']
                revision_id = revision['revid']

                # Check if wikitext was returned
                # This is necessary since when multiple revisions of the same page are requested
                # from the MediaWiki API they do not get filtered from the query response.
                if '*' not in revision:
                    e = f'Response from MediaWiki API for revision ID {revision_id} is missing ' \
                        f'the wikitext of the article. \n ' \
                        f'This is usually because the revision has ' \
                        f'been deleted from the requested Wikipedia site. \n ' \
                        f'This revision has been skipped because of this. \n'
                    logging.warning(e)
                else:
                    wikitext = revision['*']
                    data.append((title, revision_id, wikitext))
        all_data += data

    # Only return data if we have any
    if all_data:
        titles, revisions, raw_texts = map(tuple, zip(*all_data))
        return titles, revisions, raw_texts
    else:
        return [], [], []


class Article:
    """
    Class represents a processed and ready-to-go wikipedia article. Can be fed into the next
    step of the WikiPDA pipeline.
    """

    def __init__(self, revision_id, links, language):
        self.revision_id = revision_id
        self.links = links
        self.language = language


class Preprocessor:
    """
    Object for processing articles on Wikipedia using the WikiPDA preprocessing pipeline. Each
    instance is specific to a language edition of Wikipedia, since the different language editions
    use different resources.
    """

    def __init__(self, language: str, from_disk=False):
        """
        Constructor which loads a lot of resources for the processing of articles. It takes a couple
        of seconds to load everything.

        :param language: Code for the language edition of wikipedia that you wish to process
        resources for. E.g: en (english), it (italian) or sv (swedish)
        :param from_disk: Flag to choose between loading language resources from disk or using
        dictionary in RAM. Option provided to save RAM for API.
        """

        # Set the language for the articles
        self.language = language


        # Used when mapping links in the Wikitext to their underlying QIDs
        if from_disk:
            self.title_qid_mapping = FastSqliteDict(
                RESOURCE_PATH + language + '/title_qid_mappings.sqlite')
        else:
            with open(RESOURCE_PATH + language + '/title_qid_mappings.pickle', 'rb') as f:
                self.title_qid_mapping = pickle.load(f)

        # Used when densifying articles with non-existing links
        if from_disk:
            self.anchors = FastSqliteDict(RESOURCE_PATH + language + '/qid_mappings.sqlite')
        else:
            with open(RESOURCE_PATH + language + '/qid_mappings.pickle', 'rb') as f:
                self.anchors = pickle.load(f)

        # Matrix factorization used for disambiguating ambiguous anchors
        # NOTE: these are accessed from disk using memmap
        self.users = np.load(RESOURCE_PATH + language + '/users.npy', mmap_mode='r')
        self.products = np.load(RESOURCE_PATH + language + '/products.npy', mmap_mode='r')

        # This contains the matrix indices in the factorization for articles
        # I.e., mapping from QID -> Index in matrix factorization
        if from_disk:
            self.matrix_positions = FastSqliteDict(RESOURCE_PATH + language + '/matrix_positions.sqlite')
        else:
            with open(RESOURCE_PATH + language + '/matrix_positions.pickle', 'rb') as f:
                self.matrix_positions = pickle.load(f)

    @staticmethod
    def extract_links(raw_texts: List[str], title_qid_mapping: Dict) \
            -> Tuple[List[List[str]], List[List[str]]]:
        """
        Takes a given wikitext, extracts the links and then parses that text.

        :param raw_texts: The wikitext representation of Wikipedia articles.
        :param title_qid_mapping: Mapping between the titles and QIDs of Wikipedia articles.
        :return: The QIDs and texts without wikicode.
        """

        links_regex = re.compile(
            r"\[\[(?P<link>[^\n\|\]\[\<\>\{\}]{0,256})(?:\|(?P<anchor>[^\[]*?))?\]\]")

        references_regex = re.compile(r"<ref[^>]*>[^<]+<\/ref>")

        filtered_texts = []
        links = []
        for text in raw_texts:
            # Extract links
            links.append([match[0] for match in links_regex.findall(text)])

            # Remove links from plain text
            filtered = re.sub(links_regex, '.', text)
            # Remove references
            filtered = re.sub(references_regex, '.', filtered)
            
            filtered_texts.append(filtered.lower())
        
        # Parse and retrieve plaintext of articles without wikicode
        clean_texts = [mw.parse(text).strip_code() for text in filtered_texts]

        # The links are currently represented as the titles of the articles
        # Iterate over them and retrieve their QIDs instead
        qid_links = []
        for link_set in links:
            qid_links_article = []
            for title in link_set:
                if title is not None and title != '':
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
    def create_embedding(products: np.array, existing_links: List[str], matrix_indices: dict) \
            -> np.array:
        """
        Computes embedding vector u for given links using the latent embeddings for the product
        matrix V. It is computed using the dot product against V. Sparse matrices are used to
        increase efficiency of matrix product.

        :param products: The vector embeddings for the candidate articles in a matrix
        factorization model
        :param existing_links: The links that were already present in the document explicitly
        :param matrix_indices: Mappings between qids and matrix indices for matrix factorization
        :return: Computed embedding u
        """

        # Compute adjacency vector
        N, M = products.shape
        a = OrderedDict()
        dense_i = OrderedDict()
        for link in existing_links:
            index = matrix_indices.get(link)
            if index is not None:
                # Increment adjacency count
                if index not in a:
                    a[index] = 1
                else:
                    a[index] += 1

                # Keep track of all dense entries
                dense_i[index] = 1

        # Create sparse matrix using only dense rows
        dense_i = list(dense_i.keys())
        dense = products[dense_i]

        # Compute embedding using dense product
        a = np.array(list(a.values()))
        u = a @ dense
        return u.flatten()

    @staticmethod
    def disambiguate_phrases(qids: List[List[str]], u: np.array, products: np.array,
                             matrix_indices: dict) -> List[str]:
        """
        Selects the best candidate for a set of phrases. Expects there to be more than one set
        of links to disambiguate - meaning that multiple anchors can be disambiguated
        simultaneously.

        :param qids: The candidate anchors for all phrases to process.
        :param u: Latent embedding of source article
        :param products: The vector embeddings for the candidate articles in a matrix
        factorization model
        :param matrix_indices: Mappings between qids and matrix indices for matrix factorization
        :return: The chosen links
        """

        # Find the candidate links
        candidates_ensemble = []
        for candidates in qids:
            cs = dict()
            for candidate in candidates:

                # Skip if candidate article not in matrix factorization
                value = matrix_indices.get(candidate)
                if value:
                    cs[candidate] = value

            candidates_ensemble.append(cs)

        all_indices = np.array([index for cs in candidates_ensemble for index in cs.values()])

        # Only access the rows of interest once to save time on loading from disk
        unique_indices = np.unique(all_indices)
        candidate_products = products[unique_indices]
        unique_scores = u @ candidate_products.T
        unique_scores = {m_index: unique_scores[i] for i, m_index in enumerate(unique_indices)}

        scores = [unique_scores[i] for i in all_indices]

        # Select best candidate for each phrase
        best_candidates = []
        start_index = 0
        for cs in candidates_ensemble:
            candidates = list(cs.keys())
            phrase_scores = scores[start_index:start_index+len(candidates)]
            best_candidates.append(candidates[np.argmax(phrase_scores)])
        return best_candidates

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
        mi = None
        if wikidata_id is not None:
            mi = matrix_indices.get(wikidata_id)

        # Get embedding if already present
        if mi is not None:
            u = users[mi]

        # Otherwise compute embedding by projecting the links onto V
        # We only project it onto the columns of V which will not be 0 since
        # our link vector will be sparse
        else:
            u = Preprocessor.create_embedding(products, existing_links, matrix_indices)

        ambiguous_anchors = []
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

                    # Phrase needs to be disambiguated
                    elif len(qids) > 1:
                        ambiguous_anchors.append(qids)

                    # Append found non-ambiguous anchor
                    elif qids is not None:
                        found_anchors.append(qids[0])

                    # Remove phrase from text
                    chunk.replace(ng, " @@ ")

        # Disambiguate phrases in bulk
        if ambiguous_anchors:
            found_anchors += Preprocessor.disambiguate_phrases(ambiguous_anchors, u,
                                                               products, matrix_indices)

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
        qid_links, clean_texts = Preprocessor.extract_links(wikitexts, self.title_qid_mapping)

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

            # Article is ready
            revision = revisions[i] if revisions is not None else 'NA'
            articles.append(Article(revision, qid_links[i], self.language))

        return articles


class Bagger:

    def __init__(self):
        with open(DATA_DIR + 'topic_dict.pkl', 'rb') as f:
            self.topic_dictionary = pickle.load(f)

    def bag(self, articles: List[Article]):
        bols = []
        for article in articles:
            bols.append(self.topic_dictionary.doc2bow(article.links))

        return bols


class FastSqliteDict:

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.c = self.conn.cursor()
        self.c.execute('PRAGMA cache_size = -10000')
        self.c.execute('PRAGMA journal_mode = OFF')
        self.c.execute('PRAGMA synchronous = OFF')
        self.c.execute('PRAGMA locking_mode = EXCLUSIVE')

    def __getitem__(self, key):
        value = self.c.execute(f'SELECT value FROM unnamed WHERE key=?', (key,)).fetchone()
        return pickle.loads(value[0]) if value else value

    def get(self, key):
        return self.__getitem__(key)
