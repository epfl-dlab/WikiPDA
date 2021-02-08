"""
Module contains configuration parameters for Swagger (API documentation) to keep other modules
clean.
"""

SWAGGER_CONFIG = {
  "swagger": "2.0",
  "info": {
    "title": "WikiPDA API",
    "description": "API that enables getting WikiPDA "
                   "embeddings for articles without installing the library.",
    "contact": {
      "responsibleOrganization": "Data Science laboratory EPFL (dlab)",
      "responsibleDeveloper": "Daniel Berg Thomsen",
      "email": "danielbergthomsen@gmail.com",
    },
    # "termsOfService": "http://me.com/terms",
    "version": "1.0"
  },
  # "host": "mysite.com",  # overrides localhost:5000
  "schemes": [
    "http",
    "https"
  ],
}
