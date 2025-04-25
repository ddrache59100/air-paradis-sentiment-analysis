# application.py

from api import app, load_model_and_resources

# Initialiser le modèle
import api
api.model, api.tokenizer, api.model_type, api.current_config = load_model_and_resources()
