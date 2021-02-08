"""
Module contains the setup and configuration of the API itself.
"""

from flask import Flask, Blueprint
from flask_restful import Api
from flasgger import Swagger
from resources.embedding import TopicEmbeddingsRevision, TopicEmbeddingsWikitext
from resources.text_category import CategoryPredictionsRevision, CategoryPredictionsWikitext
from resources.topic_distribution import TopicDistribution
from common.util import RevisionListConverter
from swagger_config import SWAGGER_CONFIG
from flask_config import WikiPDAConfig


app = Flask(__name__)
swagger = Swagger(app, template=SWAGGER_CONFIG)  # configures API documentation
app.config.from_object(WikiPDAConfig)

# Add list converter for multiple resource IDs
# This allows us to provide a the resource IDs separated by '|' like in MediaWiki API
app.url_map.converters['list'] = RevisionListConverter

# configure flask-restful
api_bp = Blueprint('api', __name__)
api = Api(api_bp)
api.add_resource(TopicEmbeddingsRevision, '/topic_embedding/<list:revids>')
api.add_resource(TopicEmbeddingsWikitext, '/topic_embedding/')
api.add_resource(CategoryPredictionsRevision, '/text_category/<list:revids>')
api.add_resource(CategoryPredictionsWikitext, '/text_category/')
api.add_resource(TopicDistribution, '/topic_distribution/<int:k>')
app.register_blueprint(api_bp)

# This is for when debugging the API. To deploy it look at the instructions in the README
if __name__ == '__main__':
    app.run(debug=True)
