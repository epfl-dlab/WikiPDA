"""
Module contains the classes that allow one to produce the topic embeddings and text categories of
articles run through the WikiPDA pre-processing pipeline.
"""

import os
from wikipda.settings import CLASSIFIERS_PATH, LDA_MODEL_PATH
from wikipda.article import Article
from typing import List

from gensim.models.ldamulticore import LdaMulticore
import numpy as np
from xgboost import XGBClassifier


class TextClassifier:
    """
    Class allows for computing the text category of topic embeddings. 
    NOTE: there is currently only a classifier trained on embeddings with k=300.
    """

    def __init__(self):

        # attempt to load model
        path = CLASSIFIERS_PATH
        self.models = []
        self.text_categories = []
        self._load_models(path)

    def _load_models(self, path: str):
        """
        Loads all the different classifiers for the various text categories (64 in total)

        :param path: Path to the folder containing all the trained classifiers.
        """

        # verify that model is present in directory
        if not os.path.exists(path):
            e = 'Classifiers not present in data folder. Make sure you have downloaded them from ' \
                'the drive folder!'
            raise FileNotFoundError(e)

        # load models
        for model_filename in os.listdir(path):
            bst = XGBClassifier()
            bst.load_model(path + model_filename)
            text_category = os.path.splitext(model_filename)[0]
            self.models.append(bst)
            self.text_categories.append(text_category)

    def predict(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Produce category predictions for the given topic embeddings by running them through all
        the models and picking the one who is the most confident in their prediction.

        :param embeddings: The topic embeddings produced using the LDAModel class
        :return: the one-hot encoded predictions for the category of the embeddings
        """
        probabilities = self.predict_proba(embeddings)
        predictions = np.zeros(probabilities.shape)
        for i, y in enumerate(probabilities):
            selected_index = np.argmax(y)
            predictions[i, selected_index] = 1  # set prediction to highest confidence prediction

        return predictions

    def predict_category(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Produce category predictions and also return the predictions in plain text.
        E.g.: 'STEM.Pjhysics' etc.

        :param embeddings: The topic embeddings produced using the LDAModel class
        :return: The plain-text predicted text categories of the embeddings
        """
        predictions = self.predict(embeddings)
        categories = np.empty((len(predictions)), dtype=object)
        for i, prediction in enumerate(predictions):
            selected_index = np.argmax(prediction)
            categories[i] = self.text_categories[selected_index]

        return categories

    def predict_proba(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Produce the category predictions in terms of probabilities on the given topic embeddings.

        :param embeddings: The topic embeddings produced using the LDAModel class
        :return: The probability matrix of the category predictions for the embeddings.
        """
        X = np.array(embeddings)
        if len(X.shape) > 2:
            X = X[:, :, 1]

        probas = np.zeros((len(X), len(self.models)))
        for i, model in enumerate(self.models):
            probas[:, i] = model.predict_proba(X)[:, 1]  # Select only probability of positive class
        return probas


class LDAModel:
    """
    Class for producing the embeddings for an article which has been run through the WikiPDA
    pre-processing pipeline.
    """

    def __init__(self, k: int = 300):
        """
        Pretty standard constructor which attempts to load the given model from the resource
        directory.
        :param k: The parameter k of the pre-trained LDA-model that you wish to produce embeddings
        from
        """

        # Attempt to load model
        path = LDA_MODEL_PATH + str(k)
        self.k = k
        self.model = None
        self._load_model(path)

    def _load_model(self, path: str):
        """
        Loads the given LDA model contained in the given path.

        :param path: The folder in which the model should be contained
        """

        # Verify that model is present in directory
        if not os.path.exists(path):
            e = f'Model not present in {path} for configured k. Make sure you have downloaded ' \
                f'the model using the ResourceDownloader class'
            raise FileNotFoundError(e)

        self.model = LdaMulticore.load(path + '/lda.model')

    def get_embeddings(self, articles: List[Article]) -> List[np.ndarray]:
        """
        Produces topic embeddings for the given articles that have been run through the entire
        WikiPDA preprocessing pipeline.

        :param articles: Article class instances to produce topic embeddings for
        :return: The topic embeddings for the given Article instances.
        """
        embeddings = []
        for bol in [article.bol for article in articles]:
            embedding = self.model.get_document_topics(bol, minimum_probability=0)
            embeddings.append(embedding)
        return embeddings

    def get_topics(self, num_links: int) -> np.ndarray:
        """
        Gives the topic distributions of the underlying LDA model.

        :param num_links: How many of the most probable links to retrieve for each topic.
        :return: Topic distributions of the model
        """
        return self.model.show_topics(self.k, num_links, formatted=False)
