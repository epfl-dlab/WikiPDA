# WikiPDA API
This is the folder for an API that produces cross-lingual topic embeddings using 
[WikiPDA](https://arxiv.org/abs/2009.11207). The goal of the API is to provide an easy way
to get a feel for the technique without having to install the libraries.

## Development setup
1. Clone the repository.
2. Install the dependencies (preferably in a virtual environment) using `pip install -r requirements.txt`.
3. Set up the WikiPDA library. Instructions available [here](https://github.com/epfl-dlab/WikiPDA/tree/master/WikiPDA-Lib).
4. Configure `settings.py` to setup the languages and LDA models you wish to support.
5. Run `python api.py` in order to start the flask server.
6. Go to the Swagger UI to see that everything is working: 
`http://127.0.0.1:5000/apidocs`

## Deploying
1. Make sure the Swagger documentations is up-to-date (look in particular at version and host).
2. Insert multiple steps here depending on where we decide to deploy it.
3. ...
4. Profit!

For now one can run `gunicorn api:app -w 1` to see what things looks like in production.