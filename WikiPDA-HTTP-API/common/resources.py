"""
This module contains the resources used in common between the different flask-resftul resource
definitions.
"""

from wikipda.article import Preprocessor
from wikipda.model import LDAModel, TextClassifier
from settings import SUPPORTED_LANGUAGES, SUPPORTED_LDA

PREPROCESSORS = {lang: Preprocessor(lang) for lang in SUPPORTED_LANGUAGES}
LDA_MODELS = {k: LDAModel(k) for k in SUPPORTED_LDA}
TEXT_CLASSIFIER = TextClassifier()
