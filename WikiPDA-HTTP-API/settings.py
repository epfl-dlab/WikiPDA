"""
Module contains settings for the API related to the amount of resources exposed.
"""

# This configures the language resources accepted in the API
# NOTE: you should make sure that you have downloaded the language-specific resources from
# the drive folder before entering a language here.
# The entries themselves are the string representation of the langauge code for a given
# language project.
SUPPORTED_LANGUAGES = ['en']

# The LDA models supported (the entries represent the k configured for the given model).
# Again these need to be downloaded from the drive folder.
SUPPORTED_LDA = [300]
