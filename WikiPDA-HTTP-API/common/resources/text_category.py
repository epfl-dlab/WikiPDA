"""
This module contains the flask-restful resource definition for text-categories of different
Wikipedia articles. It essentially does the same thing as embedding.py but also uses the
TextClassifier module of the library to also produce a category predictions for the given article.
"""

from flask import request
from flask_restful import Resource, reqparse, inputs
from settings import SUPPORTED_LANGUAGES
from common.resources import PREPROCESSORS, LDA_MODELS, TEXT_CLASSIFIER

from wikipda.article import fetch_article_data


class CategoryPredictionsRevision(Resource):

    def get(self, revids):
        """
        Endpoint allows getting the text category prediction for a given wikipedia article,
        specified using the revision ID. NOTE: only available for the LDA model with the highest
        k configured (k=300).
            ---
            parameters:
              - name: revids
                description: wikipedia revision ids
                in: path
                type: string
                required: true
                examples:
                  albert einstein:
                    value: 998444189
                    summary: The Albert Einstein article

              - name: lang
                description: language edition of wikipedia
                in: query
                type: string
                required: true
                example: en

              - name: enrich
                description: Whether to perform link densification on text or not (recommended)
                in: query
                type: bool
                default: True

            responses:
              200:
                description: The topic embeddings for the given article revisions
                schema:
                  id: CategoryPredictions
                  type: object
                  properties:
                    text_categories:
                      type: array
                      items:
                        type: string
       """
        parser = reqparse.RequestParser()
        parser.add_argument('lang',
                            choices=SUPPORTED_LANGUAGES,
                            type=str,
                            required=True,
                            help='Unknown language: {error_msg}')

        parser.add_argument('enrich',
                            default=True,
                            type=inputs.boolean)

        args = parser.parse_args()

        # Load and process articles
        titles, revisions, wikitexts = fetch_article_data(revids, args.lang)
        articles = PREPROCESSORS[args.lang].load(wikitexts, revisions, titles, enrich=args.enrich)

        # Load model
        model = LDA_MODELS[300]

        # Produce embeddings
        embeddings = model.get_embeddings(articles)

        # Produce category predictions
        text_categories = TEXT_CLASSIFIER.predict_category(embeddings)
        return {'text_categories': text_categories.tolist()}


class CategoryPredictionsWikitext(Resource):

    def post(self):
        """
        Endpoint allows getting the text category prediction for a given set of Wikitexts.
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

              - name: enrich
                description: Whether to perform link densification on text or not (recommended)
                in: query
                type: bool
                default: True

            responses:
              200:
                description: The topic embeddings for the given article revisions
                schema:
                  id: TextCategories
                  type: object
                  properties:
                    text_categories:
                      type: array
                      items:
                        type: string
       """

        parser = reqparse.RequestParser()
        parser.add_argument('lang',
                            choices=SUPPORTED_LANGUAGES,
                            type=str,
                            required=True,
                            help='Unknown language: {error_msg}')

        parser.add_argument('enrich',
                            default=True,
                            type=inputs.boolean)

        args = parser.parse_args()

        # User should provide the Wikitexts they want to have processed in request
        wikitexts = request.json['wikitexts']

        # Load and process articles
        articles = PREPROCESSORS[args.lang].load(wikitexts, enrich=args.enrich)

        # Load model
        model = LDA_MODELS[300]

        # Produce embeddings
        embeddings = model.get_embeddings(articles)

        # Produce category predictions
        text_categories = TEXT_CLASSIFIER.predict_category(embeddings)
        return {'text_categories': text_categories.tolist()}

