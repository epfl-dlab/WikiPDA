"""
This module contains the resource instances used in common between the different flask-restful
resource definitions.
"""

from wikipda.article import Preprocessor, Bagger
from wikipda.model import WikiPDAModel, ORESClassifier
from settings import SUPPORTED_LANGUAGES, SUPPORTED_LDA

PREPROCESSORS = {lang: Preprocessor(lang, from_disk=True) for lang in SUPPORTED_LANGUAGES}
BAGGER = Bagger()
LDA_MODELS = {k: WikiPDAModel(k) for k in SUPPORTED_LDA}
TEXT_CLASSIFIER = ORESClassifier()
