"""
Module contains the setup and configuration of the API itself.
"""

from flask import Flask, Blueprint
from flask_restful import Api
from flasgger import Swagger
from common.resources.topic_distribution import TopicsDistributionRevision, TopicsDistributionWikitext
from common.resources.text_category import CategoryPredictionsRevision, CategoryPredictionsWikitext
from common.resources.topics_qid import TopicsQID
from common.util import RevisionListConverter
from swagger_config import SWAGGER_CONFIG
from flask_config import WikiPDAConfig


app = Flask(__name__)
swagger = Swagger(app, template=SWAGGER_CONFIG)  # Configures API documentation
app.config.from_object(WikiPDAConfig)

# Add list converter for multiple resource IDs
# This allows us to provide a the resource IDs separated by '|' like in MediaWiki API
app.url_map.converters['list'] = RevisionListConverter

# Configure flask-restful
api_bp = Blueprint('api', __name__)
api = Api(api_bp)
api.add_resource(TopicsDistributionRevision, '/topics_distribution/<list:revids>')
api.add_resource(TopicsDistributionWikitext, '/topics_distribution/')
api.add_resource(CategoryPredictionsRevision, '/predict_labels/<list:revids>')
api.add_resource(CategoryPredictionsWikitext, '/predict_labels/')
api.add_resource(TopicsQID, '/topics_qids/<int:k>')
app.register_blueprint(api_bp)

# This is for when debugging the API. To deploy it look at the instructions in the README
if __name__ == '__main__':
    app.run(debug=True)
