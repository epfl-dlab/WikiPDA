"""
Module contains the flask-restful definition for the topic embedding resource.
This resource takes the articles and processes them through the WikiPDA pipeline using the
WikiPDA library.
"""

from flask import request, abort, make_response, jsonify
from flask_restful import Resource, reqparse, inputs
from settings import SUPPORTED_LANGUAGES, SUPPORTED_LDA
from common.resource_instances import PREPROCESSORS, BAGGER, LDA_MODELS
from functools import lru_cache
from wikipda.article import fetch_article_data


class TopicsDistributionRevision(Resource):
    @lru_cache(maxsize=256)
    def get(self, revids):
        """
        Returns the topic distribution for a provided set of Wikipedia revisions.
        Each request is limited to a maximum of 15 different articles at a time.
            ---
            parameters:
              - name: revids
                description: Wikipedia revision ids, separated using the | token (max 15).
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

        # Make sure users do not hog too many resources.
        if len(revids) > 15:
            response = make_response(jsonify(message="Passed too many articles, max is 15."), 431)
            abort(response)

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

        # Bag the link vectors
        bols = BAGGER.bag(articles)

        # Load model
        model = LDA_MODELS[args.dimensions]

        # Produce embeddings
        embeddings = model.get_embeddings(bols)
        return {'topics_distribution': embeddings}


class TopicsDistributionWikitext(Resource):
    @lru_cache(maxsize=256)
    def post(self):
        """
        Returns the topic distribution for a provided set of Wikicode documents.
        Each request is limited to a maximum of 15 different articles at a time.
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

        # Make sure users do not hog too many resources.
        if len(wikitexts) > 15:
            response = make_response(jsonify(message="Passed too many articles, max is 15."), 431)
            abort(response)

        # Load and process articles
        articles = PREPROCESSORS[args.lang].load(wikitexts, enrich=args.enrich)

        # Bag the link vectors
        bols = BAGGER.bag(articles)

        # Load model
        model = LDA_MODELS[args.dimensions]

        # Produce embeddings
        embeddings = model.get_embeddings(bols)
        return {'topics_distribution': embeddings}
