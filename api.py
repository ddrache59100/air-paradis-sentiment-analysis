# api.py avec gestion amelioree des modeles

import os
import re
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify

model = None
tokenizer = None
model_type = None
current_config = None

# Importations conditionnelles selon le type de modele
try:
    import tensorflow as tf
    tf_available = True

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Ne configure que pour le premier GPU visible si necessaire
            # Ou configure pour tous les GPUs detectes
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Croissance memoire activee pour les GPUs: {gpus}")
        except RuntimeError as e:
            # La croissance memoire doit etre activee avant l'initialisation des GPUs
            print(f"Erreur lors de l'activation de la croissance memoire: {e}")


except ImportError:
    tf_available = False

try:
    from transformers import TFBertForSequenceClassification, BertTokenizer
    from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
    transformers_available = True
except ImportError:
    transformers_available = False

# --- Initialisation de l'application Flask ---
app = Flask(__name__)

# Configuration pour permettre de specifier un fichier de configuration alternatif
app.config.from_mapping(
    CUSTOM_CONFIG_PATH=None  # Par defaut, aucun fichier de configuration personnalise
)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.json')

def load_model_config():
    """
    Charge la configuration du modèle à partir du fichier de configuration.
    Utilise la configuration personnalisée si spécifiée, sinon utilise la configuration par défaut.
    """
    # Vérifier si un chemin de configuration personnalisé est spécifié
    custom_config_path = app.config.get('CUSTOM_CONFIG_PATH')
    
    # Utiliser le chemin de configuration personnalisé s'il est spécifié et existe
    if custom_config_path and os.path.exists(custom_config_path):
        config_path = custom_config_path
        print(f"Utilisation de la configuration personnalisée: {config_path}")
    else:
        config_path = DEFAULT_CONFIG_PATH
        print(f"Utilisation de la configuration par défaut: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            model_paths = json.load(f)
        
        # Résoudre les chemins relatifs par rapport au répertoire de base
        if model_paths.get("best_model") and not os.path.isabs(model_paths.get("best_model")):
            model_paths["best_model"] = os.path.join(BASE_DIR, model_paths["best_model"])
        
        if model_paths.get("tokenizer") and model_paths.get("tokenizer") is not None and not os.path.isabs(model_paths.get("tokenizer")):
            model_paths["tokenizer"] = os.path.join(BASE_DIR, model_paths["tokenizer"])
        
        return model_paths
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement de la configuration depuis {config_path}: {e}")
        # Configuration par défaut si fichier non trouvé ou invalide
        return {
            "best_model": os.path.join(BASE_DIR, 'models', 'classical', 'regression_logistique.pkl'),
            "best_model_type": "classical",
            "tokenizer": None
        }
        
# --- Fonctions de nettoyage de texte ---
def clean_text(text):
    """
    Nettoie un texte en supprimant les URLs, mentions, hashtags et caracteres speciaux.
    """
    if not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Suppression des mentions @utilisateur
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Suppression des hashtags (on garde le mot mais pas le #)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    
    # Suppression des caracteres non alphanumeriques
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Chargement du modele et des ressources ---
def load_model_and_resources():
    """
    Charge le modele et les ressources associees en fonction du type de modele.
    """
    # Charger la configuration
    model_paths = load_model_config()
    
    best_model_path = model_paths.get("best_model")
    best_model_type = model_paths.get("best_model_type", "classical")
    tokenizer_path = model_paths.get("tokenizer")
    
    model = None
    tokenizer = None
    model_type = None
    current_config = None
    
    try:
        print(f"Chargement du modele {best_model_type} depuis: {best_model_path}")
        
        # Verifier que le fichier du modele existe
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Le fichier modele n'existe pas: {best_model_path}")
        
        if best_model_type == 'classical':
            try:
                with open(best_model_path, 'rb') as f:
                    model = pickle.load(f)
                print("Modele classique charge avec succes")
                
                # Verifier si le modele a les methodes necessaires
                has_predict = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
                has_predict_proba = hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))
                
                print(f"Le modele a une methode 'predict': {has_predict}")
                print(f"Le modele a une methode 'predict_proba': {has_predict_proba}")
                
                if not has_predict:
                    raise ValueError("Le modele n'a pas de methode 'predict'")
                
            except (pickle.PickleError, ImportError, AttributeError) as e:
                raise RuntimeError(f"Erreur lors du chargement du modele pickle: {str(e)}. "
                                  f"Cela peut etre dû a un format incompatible ou a une version differente de scikit-learn.")
            
        elif best_model_type == 'deeplearning':
            if not tf_available:
                raise ImportError("TensorFlow n'est pas disponible, impossible de charger un modele deep learning")
            model = tf.keras.models.load_model(best_model_path)
            
            if tokenizer_path is None:
                raise ValueError("Le chemin du tokenizer est requis pour un modele deep learning")
            
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Le fichier tokenizer n'existe pas: {tokenizer_path}")
                
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print("Modele deep learning et tokenizer charges avec succes")
            
        elif best_model_type == 'transformer':
            if not transformers_available:
                raise ImportError("La bibliotheque transformers n'est pas disponible")
                
            if 'distilbert' in best_model_path.lower():
                model = TFDistilBertForSequenceClassification.from_pretrained(best_model_path)
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                print("Modele DistilBERT et tokenizer charges avec succes")
            else:
                model = TFBertForSequenceClassification.from_pretrained(best_model_path)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                print("Modele BERT et tokenizer charges avec succes")
                
        else:
            raise ValueError(f"Type de modele non reconnu: {best_model_type}")
            
        return model, tokenizer, best_model_type, model_paths
        
    except Exception as e:
        print(f"Erreur lors du chargement du modele: {e}")
        import traceback
        traceback.print_exc()
        
        # Au lieu de quitter l'application, lever l'exception pour qu'elle puisse etre geree
        raise

# --- Variable globale pour stocker la configuration utilisee ---
current_config = None

# --- Fonction de prediction selon le type de modele ---
def predict_with_model(text, model_type, model, tokenizer=None):
    """
    Predit le sentiment d'un texte selon le type de modele.
    """
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        # Texte vide apres nettoyage
        return "Negatif", 0, {"Negatif": 1.0, "Positif": 0.0}, cleaned_text
    
    if model_type == 'classical':
        # Pour les modeles classiques (avec pipeline sklearn)
        try:
            # Prediction de la classe
            prediction_class = model.predict([cleaned_text])[0]
            
            # Verifier si le modele dispose de predict_proba
            if hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')):
                # Utiliser predict_proba si disponible
                probas = model.predict_proba([cleaned_text])[0]
                probabilities = {"Negatif": float(probas[0]), "Positif": float(probas[1])}
            else:
                # Sinon, utiliser une approximation basee sur la classe predite
                probabilities = {
                    "Negatif": float(1.0 - prediction_class),
                    "Positif": float(prediction_class)
                }
            
            sentiment = "Positif" if prediction_class == 1 else "Negatif"
        
        except Exception as e:
            print(f"Erreur lors de la prediction avec le modele classique: {e}")
            import traceback
            traceback.print_exc()
            # Prediction par defaut en cas d'erreur
            sentiment = "Negatif"
            prediction_class = 0
            probabilities = {"Negatif": 1.0, "Positif": 0.0}
        
    elif model_type == 'deeplearning':
        # Pour les modeles deep learning
        # Tokenisation et padding
        sequences = tokenizer.texts_to_sequences([cleaned_text])
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded = pad_sequences(sequences, maxlen=100)  # Ajustez maxlen si necessaire
        
        # Prediction
        prediction = model.predict(padded)
        sentiment_score = float(prediction[0][0])
        prediction_class = 1 if sentiment_score > 0.5 else 0
        sentiment = "Positif" if prediction_class == 1 else "Negatif"
        probabilities = {"Negatif": float(1 - sentiment_score), "Positif": float(sentiment_score)}
        
    elif model_type == 'transformer':
        # Pour les modeles BERT/DistilBERT
        # Tokenisation
        inputs = tokenizer(cleaned_text, return_tensors="tf", padding=True, truncation=True)
        
        # Prediction
        outputs = model(inputs)
        logits = outputs.logits.numpy()
        import tensorflow as tf
        probas = tf.nn.softmax(logits, axis=1).numpy()[0]
        prediction_class = np.argmax(logits, axis=1)[0]
        sentiment = "Positif" if prediction_class == 1 else "Negatif"
        probabilities = {"Negatif": float(probas[0]), "Positif": float(probas[1])}
    
    return sentiment, int(prediction_class), probabilities, cleaned_text

# --- Endpoint pour changer la configuration ---
@app.route('/config', methods=['POST'])
def change_config():
    if not request.is_json:
        return jsonify({"error": "La requete doit etre au format JSON"}), 400
    
    data = request.get_json()
    config_path = data.get('config_path')
    
    if not config_path:
        return jsonify({"error": "Le chemin du fichier de configuration est requis"}), 400
    
    # Verifier si le chemin est valide
    if not os.path.exists(config_path):
        return jsonify({"error": f"Le fichier de configuration '{config_path}' n'existe pas"}), 404
    
    try:
        # Recharger le modele avec la nouvelle configuration
        _, _, model_type, config = reload_model_with_config(config_path)
        
        return jsonify({
            "success": True,
            "message": f"Configuration modifiee avec succes: {config_path}",
            "model_type": model_type,
            "config": config
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Erreur lors du changement de configuration: {str(e)}"
        }), 500

# --- Endpoint pour verifier la configuration actuelle ---
@app.route('/config', methods=['GET'])
def get_current_config():
    return jsonify({
        "config_path": app.config.get('CUSTOM_CONFIG_PATH') or DEFAULT_CONFIG_PATH,
        "is_custom": app.config.get('CUSTOM_CONFIG_PATH') is not None,
        "config": current_config,
        "model_info": {
            "has_predict": hasattr(model, 'predict') and callable(getattr(model, 'predict')),
            "has_predict_proba": hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')),
            "type": str(type(model).__name__)
        }
    })

# --- Endpoint de prediction ---
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if not request.is_json:
        return jsonify({"error": "La requete doit etre au format JSON"}), 400

    data = request.get_json()
    text = data.get('text', None)

    if text is None or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Le champ 'text' est manquant, vide ou n'est pas une chaine de caracteres"}), 400

    try:
        # Verifier si le modele est charge
        if model is None:
            return jsonify({
                "error": "Aucun modele n'est charge. Impossible de faire une prediction.",
                "suggestion": "Verifiez les logs du serveur et assurez-vous que le modele est correctement configure."
            }), 500
            
        # Prediction
        sentiment, prediction, probabilities, cleaned_text = predict_with_model(
            text, model_type, model, tokenizer
        )
        
        # Resultat
        return jsonify({
            "input_text": text,
            "cleaned_text": cleaned_text,
            "label": sentiment,
            "prediction": prediction,
            "probabilities": probabilities,
            "model_type": model_type
        })
    
    except Exception as e:
        print(f"Erreur lors de la prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Une erreur interne est survenue lors de la prediction.", 
                       "details": str(e)}), 500

# --- Endpoint de statut ---
@app.route('/status', methods=['GET'])
def status():
    global model, model_type, current_config  # Référencez explicitement les variables globales
    
    return jsonify({
        "status": "API operationnelle",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None if model_type != 'classical' else True,
        "model_type": model_type,
        "model_path": current_config.get("best_model") if current_config else None,
        "config_path": app.config.get('CUSTOM_CONFIG_PATH') or DEFAULT_CONFIG_PATH,
        "is_custom_config": app.config.get('CUSTOM_CONFIG_PATH') is not None,
        "model_info": {
            "has_predict": hasattr(model, 'predict') and callable(getattr(model, 'predict')) if model else False,
            "has_predict_proba": hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')) if model else False,
            "type": str(type(model).__name__) if model else None
        }
    })

# --- Endpoint d'accueil ---
@app.route('/', methods=['GET'])
def welcome():
    """
    Page d'accueil simple pour l'API.
    """
    return jsonify({
        "name": "Air Paradis - API d'analyse de sentiment",
        "description": "Cette API permet d'analyser le sentiment de tweets pour anticiper les bad buzz.",
        "version": "1.0.0",
        "endpoints": {
            "/": "Cette page d'accueil (GET)",
            "/status": "Statut de l'API et informations sur le modele charge (GET)",
            "/predict": "Analyse le sentiment d'un texte fourni (POST)",
            "/config": "Obtenir ou modifier la configuration (GET/POST)"
        },
        "usage": {
            "method": "POST",
            "endpoint": "/predict",
            "content_type": "application/json",
            "body": {
                "text": "Votre texte a analyser ici"
            },
            "response": {
                "label": "Positif/Negatif",
                "prediction": "Classe numerique (0/1)",
                "probabilities": "Probabilites pour chaque classe"
            }
        },
        "model_info": {
            "type": model_type,
            "path": current_config.get("best_model") if current_config else None,
            "details": {
                "has_predict": hasattr(model, 'predict') and callable(getattr(model, 'predict')) if model else False,
                "has_predict_proba": hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba')) if model else False,
                "type": str(type(model).__name__) if model else None
            }
        },
        "config_info": {
            "path": app.config.get('CUSTOM_CONFIG_PATH') or DEFAULT_CONFIG_PATH,
            "is_custom": app.config.get('CUSTOM_CONFIG_PATH') is not None
        }
    })

# --- Lancement du serveur ---

if __name__ == '__main__':
    # Vérifier si un paramètre de configuration personnalisé est fourni en ligne de commande
    import sys
    
    # Initialiser les variables globales
    model = None
    tokenizer = None
    model_type = None
    current_config = None
    
    try:
        # Déterminer quelle configuration utiliser
        if len(sys.argv) > 1:
            custom_config_path = sys.argv[1]
            if os.path.exists(custom_config_path):
                print(f"Utilisation de la configuration personnalisée via ligne de commande: {custom_config_path}")
                app.config['CUSTOM_CONFIG_PATH'] = custom_config_path
                
                # Charger la configuration
                try:
                    with open(custom_config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Résoudre les chemins relatifs
                    if config.get("best_model") and not os.path.isabs(config.get("best_model")):
                        config["best_model"] = os.path.join(BASE_DIR, config["best_model"])
                    
                    if config.get("tokenizer") and config.get("tokenizer") is not None and not os.path.isabs(config.get("tokenizer")):
                        config["tokenizer"] = os.path.join(BASE_DIR, config["tokenizer"])
                    
                    current_config = config
                except Exception as e:
                    print(f"Erreur lors du chargement de la configuration personnalisée: {e}")
                    print("Fallback sur la configuration par défaut...")
                    current_config = {
                        "best_model": os.path.join(BASE_DIR, 'models', 'classical', 'regression_logistique.pkl'),
                        "best_model_type": "classical",
                        "tokenizer": None
                    }
            else:
                print(f"AVERTISSEMENT: Le fichier de configuration '{custom_config_path}' spécifié en ligne de commande n'existe pas.")
                print("Utilisation de la configuration par défaut.")
                current_config = {
                    "best_model": os.path.join(BASE_DIR, 'models', 'classical', 'regression_logistique.pkl'),
                    "best_model_type": "classical",
                    "tokenizer": None
                }
        else:
            # Pas de configuration personnalisée, utiliser la configuration par défaut
            try:
                with open(DEFAULT_CONFIG_PATH, 'r') as f:
                    config = json.load(f)
                
                # Résoudre les chemins relatifs
                if config.get("best_model") and not os.path.isabs(config.get("best_model")):
                    config["best_model"] = os.path.join(BASE_DIR, config["best_model"])
                
                if config.get("tokenizer") and config.get("tokenizer") is not None and not os.path.isabs(config.get("tokenizer")):
                    config["tokenizer"] = os.path.join(BASE_DIR, config["tokenizer"])
                
                current_config = config
                print(f"Utilisation de la configuration par défaut: {DEFAULT_CONFIG_PATH}")
            except Exception as e:
                print(f"Erreur lors du chargement de la configuration par défaut: {e}")
                current_config = {
                    "best_model": os.path.join(BASE_DIR, 'models', 'classical', 'regression_logistique.pkl'),
                    "best_model_type": "classical",
                    "tokenizer": None
                }
        
        # Charger le modèle avec la configuration déterminée
        best_model_path = current_config.get("best_model")
        best_model_type = current_config.get("best_model_type", "classical")
        tokenizer_path = current_config.get("tokenizer")
        
        print(f"Chargement du modèle {best_model_type} depuis: {best_model_path}")
        
        if best_model_type == 'classical':
            with open(best_model_path, 'rb') as f:
                model = pickle.load(f)
            print("Modèle classique chargé avec succès")
            
            # Vérifier si le modèle a les méthodes nécessaires
            has_predict = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
            has_predict_proba = hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))
            
            print(f"Le modèle a une méthode 'predict': {has_predict}")
            print(f"Le modèle a une méthode 'predict_proba': {has_predict_proba}")
            
            model_type = "classical"
            
        elif best_model_type == 'deeplearning':
            if not tf_available:
                raise ImportError("TensorFlow n'est pas disponible, impossible de charger un modèle deep learning")
            model = tf.keras.models.load_model(best_model_path)
            
            if tokenizer_path is None:
                raise ValueError("Le chemin du tokenizer est requis pour un modèle deep learning")
            
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Le fichier tokenizer n'existe pas: {tokenizer_path}")
                
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print("Modèle deep learning et tokenizer chargés avec succès")
            
            model_type = "deeplearning"
            
        elif best_model_type == 'transformer':
            if not transformers_available:
                raise ImportError("La bibliothèque transformers n'est pas disponible")
                
            if 'distilbert' in best_model_path.lower():
                model = TFDistilBertForSequenceClassification.from_pretrained(best_model_path)
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                print("Modèle DistilBERT et tokenizer chargés avec succès")
            else:
                model = TFBertForSequenceClassification.from_pretrained(best_model_path)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                print("Modèle BERT et tokenizer chargés avec succès")
                
            model_type = "transformer"
                
        else:
            raise ValueError(f"Type de modèle non reconnu: {best_model_type}")
        
        print("Modèle chargé avec succès")
    
    except Exception as e:
        print(f"ERREUR lors du chargement du modèle: {e}")
        import traceback
        traceback.print_exc()
        
        print("Tentative de chargement du modèle de régression logistique par défaut...")
        
        # Configuration de secours
        current_config = {
            "best_model": os.path.join(BASE_DIR, 'models', 'classical', 'regression_logistique.pkl'),
            "best_model_type": "classical",
            "tokenizer": None
        }
        
        try:
            with open(current_config["best_model"], 'rb') as f:
                model = pickle.load(f)
            tokenizer = None
            model_type = "classical"
            print("Modèle de secours chargé avec succès")
        except Exception as fallback_error:
            print(f"ERREUR CRITIQUE: Impossible de charger le modèle de secours: {fallback_error}")
            model = None
            tokenizer = None
            model_type = "classical"
    
    # Validons que le modèle est correctement chargé avant de démarrer le serveur
    if model is None:
        print("ERREUR FATALE: Aucun modèle n'a été chargé. Impossible de démarrer l'API.")
        sys.exit(1)
        
    print(f"Configuration utilisée: {current_config}")
    print(f"Type de modèle: {model_type}")
    print("Démarrage du serveur Flask...")
    app.run(host='0.0.0.0', port=5000, debug=True)