"""
This module contains the flask restful resource definition for the n most probable links for all
the topics in a given LDA model.
"""

from flask_restful import Resource, reqparse
from common.resources import LDA_MODELS


class TopicDistribution(Resource):

    def get(self, k):
        """
        Endpoint allows getting the most probable links for a given pretrained LDA model.
            ---
            parameters:
              - name: k
                description: the k used for the pretrained model queried
                in: path
                type: int
                required: false
                default: 300

              - name: num_links
                description: the amount of link probabilities to return for the given distributions
                in: query
                type: int
                required: false
                default: 10

            responses:
              200:
                description: The link distributions for the given LDA model.
                schema:
                  id: Link distributions
                  type: array
                  items:
                    type: array
                    items:
                      oneOf:
                        - type: number
                        - type: array
                          items:
                            type: array
                            items:
                              oneOf:
                                - type: string
                                - type: string
       """
        parser = reqparse.RequestParser()
        parser.add_argument('num_links',
                            type=int,
                            required=False)

        args = parser.parse_args()

        # load model
        model = LDA_MODELS[k]

        # return topic distributions
        return model.get_topics(num_links=args.num_links)
