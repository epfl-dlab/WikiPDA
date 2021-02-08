"""
This module contains the flask-specific configurations.
"""

from common.util import NumpyEncoder


class WikiPDAConfig(object):
    RESTFUL_JSON = {
        # This is required to ensure serialization can happen with numpy types (e.g., np.float32)
        'cls': NumpyEncoder,
    }
