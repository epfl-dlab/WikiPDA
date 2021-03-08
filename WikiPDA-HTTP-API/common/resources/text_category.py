"""
This module contains the flask-restful resource definition for text-categories of different
Wikipedia articles. It essentially does the same thing as topics_qid.py but also uses the
TextClassifier module of the library to also produce a category predictions for the given article.
"""

from flask import request, abort, make_response, jsonify
from flask_restful import Resource, reqparse, inputs
from settings import SUPPORTED_LANGUAGES
from common.resource_instances import PREPROCESSORS, BAGGER, LDA_MODELS, TEXT_CLASSIFIER
from functools import lru_cache
from wikipda.article import fetch_article_data


class CategoryPredictionsRevision(Resource):
	@lru_cache(maxsize=256)
    def get(self, revids):
        """
        Predict the ORES labels for a given Wikipedia article revision using a distribution of 300
        topics learned by WikiPDA. It returns the labels sorted (descending) by their probability.
        The labels with a probability greater than 0.5 are included in the output. Each request is
        limited to a maximum of 15 different revisions at a time.
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
        model = LDA_MODELS[300]

        # Produce embeddings
        embeddings = model.get_embeddings(bols)

        # Produce category predictions
        text_categories = TEXT_CLASSIFIER.predict_proba_labeled(embeddings, threshold=0.5)
        return {'text_categories': sorted([{'label': k, 'probability': v} for k,v in text_categories[0].items()], key=lambda r: -r['probability'])}


class CategoryPredictionsWikitext(Resource):
	@lru_cache(maxsize=256)
    def post(self):
        """
        Predict the ORES labels for a given Wikipedia article revision using a distribution of 300
        topics learned by WikiPDA. It returns the labels sorted (descending) by their probability.
        The labels with a probability greater than 0.5 are included in the output. Each request is
        limited to a maximum of 15 different revisions at a time.
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

        # Make sure users do not hog too many resources.
        if len(wikitexts) > 15:
            response = make_response(jsonify(message="Passed too many articles, max is 15."), 431)
            abort(response)

        # Load and process articles
        articles = PREPROCESSORS[args.lang].load(wikitexts, enrich=args.enrich)

        # Bag the link vectors
        bols = BAGGER.bag(articles)

        # Load model
        model = LDA_MODELS[300]

        # Produce embeddings
        embeddings = model.get_embeddings(bols)

        # Produce category predictions
        text_categories = TEXT_CLASSIFIER.predict_proba_labeled(embeddings, threshold=0.5)
        return {'text_categories': sorted([{'label': k, 'probability': v} for k,v in text_categories[0].items()], key=lambda r: -r['probability'])}
