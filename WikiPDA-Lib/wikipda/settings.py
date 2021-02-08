"""
Module contains re-configurable library settings.
"""

from pathlib import Path


# These are the paths to the downloaded resources from the drive folder. You can change these
# as you wish.
DATA_DIR = str(Path.home()) + '/wikipda_data/'
TOPIC_DICT_PATH = DATA_DIR + 'topic_dict.pkl'
LDA_MODEL_PATH = DATA_DIR + 'LDA_models/'
CLASSIFIERS_PATH = DATA_DIR + 'classifiers/'
RESOURCE_PATH = DATA_DIR + 'lang/'
