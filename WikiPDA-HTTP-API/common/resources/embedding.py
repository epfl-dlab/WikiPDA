"""
Module contains the flask-restful definition for the topic embedding resource.
This resource takes the articles and processes them through the WikiPDA pipeline using the
WikiPDA library.
"""

from flask import request
from flask_restful import Resource, reqparse, inputs
from settings import SUPPORTED_LANGUAGES, SUPPORTED_LDA
from common.resource_instances import PREPROCESSORS, LDA_MODELS

from wikipda.article import fetch_article_data


class TopicEmbeddingsRevision(Resource):

    def get(self, revids):
        """
        Endpoint allows getting the topic embedding for a given wikipedia article,
        specified using the revision ID(s).
            ---
            parameters:
              - name: revids
                description: Wikipedia revision ids, separated using the | token.
                in: path
                type: string
                required: true
                examples:
                  albert einstein:
                    value: 998444189
                    summary: The Albert Einstein article

              - name: lang
                description: Language edition of wikipedia
                in: query
                type: string
                required: true
                example: en

              - name: dimensions
                description: Amount of topics to use for embedding (parameter K in LDA)
                in: query
                type: int
                default: 300

              - name: enrich
                description: Whether to perform link densification on text or not (recommended)
                in: query
                type: bool
                default: True

            responses:
              200:
                description: The topic embeddings for the given article revisions
                schema:
                  id: TopicEmbeddings
                  type: object
                  properties:
                    topic_embeddings:
                      type: array
                      items:
                        type: array
                        items:
                          type: array
                          items:
                            type: number
       """
        parser = reqparse.RequestParser()
        parser.add_argument('lang',
                            choices=SUPPORTED_LANGUAGES,
                            type=str,
                            required=True,
                            help='Unknown language: {error_msg}')

        parser.add_argument('dimensions',
                            choices=SUPPORTED_LDA,
                            type=int,
                            default=300,
                            help=f'Topic embedding size requested not available, only the following'
                                 f' are supported in the API: {str(SUPPORTED_LDA)}')

        parser.add_argument('enrich',
                            default=True,
                            type=inputs.boolean)

        args = parser.parse_args()

        # Load and process articles
        titles, revisions, wikitexts = fetch_article_data(revids, args.lang)
        articles = PREPROCESSORS[args.lang].load(wikitexts, revisions, titles, enrich=args.enrich)

        # Load model
        model = LDA_MODELS[args.dimensions]

        # Produce embeddings
        embeddings = model.get_embeddings(articles)
        return {'topic_embeddings': embeddings}


class TopicEmbeddingsWikitext(Resource):

    def post(self):
        """
        Endpoint allows getting the topic embedding for a given set of Wikitexts.
            ---
            parameters:

              - name: wikitexts
                in: body
                required: true
                schema:
                  id: Wikitext
                  required:
                    - wikitexts
                  properties:
                    wikitexts:
                      type: array
                      items:
                        type: string

              - name: lang
                description: Language edition of wikipedia
                in: query
                type: string
                required: true
                example: en

              - name: dimensions
                description: Amount of topics to use for embedding (parameter K in LDA)
                in: query
                type: int
                default: 300

              - name: enrich
                description: Whether to perform link densification on text or not (recommended)
                in: query
                type: bool
                default: True

            responses:
              200:
                description: The topic embeddings for the given article revisions
                schema:
                  id: TopicEmbeddings
                  type: object
                  properties:
                    topic_embeddings:
                      type: array
                      items:
                        type: array
                        items:
                          type: array
                          items:
                            type: number
       """

        parser = reqparse.RequestParser()
        parser.add_argument('lang',
                            choices=SUPPORTED_LANGUAGES,
                            type=str,
                            required=True,
                            help='Unknown language: {error_msg}')

        parser.add_argument('dimensions',
                            choices=SUPPORTED_LDA,
                            type=int,
                            default=300,
                            help=f'Topic embedding size requested not available, only the following'
                                 f' are supported in the API: {str(SUPPORTED_LDA)}')

        parser.add_argument('enrich',
                            default=True,
                            type=inputs.boolean)

        args = parser.parse_args()

        # User should provide the Wikitexts they want to have processed in request
        wikitexts = request.json['wikitexts']

        # Load and process articles
        articles = PREPROCESSORS[args.lang].load(wikitexts, enrich=args.enrich)

        # Load model
        model = LDA_MODELS[args.dimensions]

        # Produce embeddings
        embeddings = model.get_embeddings(articles)
        return {'topic_embeddings': embeddings}
