"""
Module contains the utility functions used in different places of the API.
"""

from werkzeug.routing import BaseConverter
from json import JSONEncoder
import numpy as np


class NumpyEncoder(JSONEncoder):
    """
    Subclasses JSONEncoder to allow for serialization of numpy datatypes. This was necessary to
    preserve all digits of np.float32 when returning results in the API.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return str(obj)
        return JSONEncoder.default(self, obj)


class RevisionListConverter(BaseConverter):
    """
    This can be used to allow for resources separated by | to be automatically converted into lists
    of resources in flask resource definition methods. E.g.:
    -User sends GET request with argument of form: id1|id2|id3...
    -The GET function receives a list of ids of form: [id1, id2, id3, ...]
    """
    def to_python(self, value):
        return [int(revision) for revision in value.split('|')]

    def to_url(self, values):
        return '|'.join(str(value) for value in values)
