# Air Paradis - Analyse de Sentiments pour l'Anticipation des Bad Buzz

![Badge GitHub Actions](https://github.com/votre-username/air-paradis-sentiment-analysis/actions/workflows/canada_central_reglog2.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Description du projet

Ce projet vise à développer une solution d'analyse de sentiments pour la compagnie aérienne fictive "Air Paradis". L'objectif est d'anticiper les bad buzz sur les réseaux sociaux en prédisant le sentiment (positif ou négatif) associé à un tweet.

La solution comprend:
- Plusieurs approches de modélisation (classiques, deep learning, BERT)
- Une API de prédiction déployée sur Azure
- Une démarche MLOps complète avec gestion des expérimentations, déploiement continu et monitoring en production

### Caractéristiques principales

- 🔍 Prédiction de sentiment (positif/négatif) à partir de tweets
- 🧠 Différentes approches de modélisation:
  - Modèles classiques: Régression Logistique, SVM, Random Forest, Naive Bayes
  - Modèles deep learning: CNN et LSTM avec embeddings entraînables ou GloVe
  - Modèles transformers: BERT et DistilBERT
- 🚀 API REST avec Flask, déployée sur Azure
- 📊 Tracking d'expérimentations avec MLFlow
- 🔄 Déploiement continu avec GitHub Actions
- 📈 Monitoring en production avec Azure Application Insights

## Installation et utilisation

### Prérequis

- Python 3.10+
- Pip (gestionnaire de paquets Python)
- Environnement virtuel Python (recommandé)

### Installation des dépendances

Le projet utilise différents environnements selon le type de modèle:

**Pour les modèles classiques:**
```bash
python -m venv venv_api_classical
source venv_api_classical/bin/activate  # Linux/Mac
# ou
.\venv_api_classical\Scripts\activate  # Windows
pip install -r requirements_classical.txt
```

**Pour les modèles de deep learning:**
```bash
python -m venv venv_api_deeplearning
source venv_api_deeplearning/bin/activate  # Linux/Mac
# ou
.\venv_api_deeplearning\Scripts\activate  # Windows
pip install -r requirements_deeplearning.txt
```

**Pour les modèles BERT:**
```bash
python -m venv venv_api_bert
source venv_api_bert/bin/activate  # Linux/Mac
# ou
.\venv_api_bert\Scripts\activate  # Windows
pip install -r requirements_bert.txt
```

### Démarrage de l'API en local

L'API peut être démarrée avec différentes configurations de modèles:

```bash
# Modèle classique (Régression Logistique)
python api.py model_config_reglog.json

# Modèle classique (Random Forest)
python api.py model_config_random_forest.json

# Modèle classique (SVM)
python api.py model_config_svm_lineaire.json

# Modèle classique (Naive Bayes)
python api.py model_config_naive_bayes.json

# Modèle deep learning (CNN avec embeddings entraînables)
python api.py model_config_cnn_embeddings.json

# Modèle deep learning (CNN avec GloVe)
python api.py model_config_cnn_glove.json

# Modèle deep learning (LSTM avec embeddings entraînables)
python api.py model_config_lstm_embeddings.json

# Modèle deep learning (LSTM avec GloVe)
python api.py model_config_lstm_glove.json

# Modèle BERT
python api.py model_config_bert.json

# Modèle DistilBERT
python api.py model_config_distilbert.json
```

### Test de l'API

Pour tester si l'API fonctionne correctement:

```bash
python test_api.py http://localhost:5000
```

## Structure du projet

```
.
├── api.py                     # Implémentation principale de l'API Flask
├── application.py             # Point d'entrée pour le déploiement
├── build.sh                   # Script d'installation des dépendances
├── model_config*.json         # Fichiers de configuration pour les différents modèles
├── README.md                  # Ce fichier
├── requirements*.txt          # Dépendances pour les différents types de modèles
├── startup.sh                 # Script de démarrage pour le déploiement
├── test_api.py                # Script de test de l'API
├── test_api_pytest.py         # Tests de l'API avec pytest
├── test_unit.py               # Tests unitaires
└── models/                    # Répertoire contenant les modèles entraînés
    ├── classical/             # Modèles classiques (scikit-learn)
    ├── deeplearning/          # Modèles de deep learning (Keras)
    └── bert/                  # Modèles BERT et DistilBERT
```

## Modèles disponibles

| Modèle | Type | Taille | Performance |
|--------|------|--------|-------------|
| Régression Logistique | Classique | ~52 Mo | Baseline |
| SVM Linéaire | Classique | ~52 Mo | Haute précision |
| Naive Bayes | Classique | ~54 Mo | Rapide |
| Random Forest | Classique | ~59 Mo | Robuste |
| CNN (embeddings entraînables) | Deep Learning | ~30 Mo | Bonne performance |
| CNN (GloVe) | Deep Learning | ~21 Mo | Meilleure généralisation |
| LSTM (embeddings entraînables) | Deep Learning | ~60 Mo | Capture de séquences |
| LSTM (GloVe) | Deep Learning | ~21 Mo | Meilleure capture contextuelle |
| BERT | Transformer | ~438 Mo | État de l'art |
| DistilBERT | Transformer | ~268 Mo | Compromis taille/performance |

## Endpoints de l'API

- `GET /`: Page d'accueil avec informations sur l'API
- `GET /status`: État de l'API et informations sur le modèle chargé
- `POST /predict`: Analyse le sentiment d'un texte fourni
  - Body: `{"text": "Votre texte à analyser"}`
  - Réponse: Sentiment, confiance et probabilités
- `POST /feedback`: Soumet un feedback sur une prédiction pour amélioration continue
  - Body: `{"text": "texte", "predicted_sentiment": "...", "actual_sentiment": "..."}`
- `GET/POST /config`: Obtenir ou modifier la configuration du modèle

## Déploiement continu

Le projet utilise GitHub Actions pour le déploiement continu vers Azure Web App. Le pipeline:

1. Clone le dépôt
2. Configure Python
3. Prépare le package de déploiement
4. Exécute les tests unitaires
5. Génère un rapport de test
6. Déploie l'application vers Azure

## Monitoring en production

Le monitoring est assuré par Azure Application Insights:
- Traces de prédictions incorrectes signalées par les utilisateurs
- Alertes en cas de nombre élevé de mauvaises prédictions
- Analyse des statistiques pour l'amélioration continue du modèle

## Développé par

MIC (Marketing Intelligence Consulting) pour Air Paradis
