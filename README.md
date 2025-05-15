# Air Paradis - Analyse de Sentiments pour l'Anticipation des Bad Buzz



## Description du projet

Ce projet vise à développer une solution d'analyse de sentiments pour la compagnie aérienne fictive "Air Paradis". L'objectif est d'anticiper les bad buzz sur les réseaux sociaux en prédisant le sentiment (positif ou négatif) associé à un tweet.

La solution comprend:
- Plusieurs approches de modélisation (classiques, deep learning, BERT)
- Une API de prédiction déployée sur Azure
- Une démarche MLOps complète avec gestion des expérimentations, déploiement continu et monitoring en production


## Modèle utilisé dans cette API
Cette API utilise spécifiquement le modèle CNN avec embeddings entraînables:
- Réseau de neurones convolutionnels avec une couche d'embeddings personnalisée
- F1 Score: X.XX sur l'ensemble de test
- Excellente performance sur les tweets courts et moyens
- Équilibre optimal entre précision et efficacité computationnelle

## API déployée
L'API est accessible à l'adresse suivante:
- URL: https://air-paradis-sentiment-api-cnn-embed2.azurewebsites.net


## Installation et utilisation en local

### Prérequis

- Python 3.10+
- Pip (gestionnaire de paquets Python)
- Environnement virtuel Python (recommandé)

### Installation des dépendances

Le projet utilise différents environnements selon le type de modèle:


**Pour les modèles de deep learning:**
```bash
python3.10 -m venv venv_api_deeplearning
source venv_api_deeplearning/bin/activate  # Linux/Mac
# ou
.\venv_api_deeplearning\Scripts\activate  # Windows
pip install -r requirements_deeplearning.txt
```


### Démarrage de l'API en local

L'API peut être démarrée avec différentes configurations de modèles:

```bash
# Modèle deep learning (CNN avec embeddings entraînables)
FLASK_APP=api.py flask run --host=0.0.0.0
```

### Test de l'API

Pour tester si l'API fonctionne correctement:

```bash
python test_api.py http://localhost:5000
[14:29:41] Test de l'API sur: http://localhost:5000
[14:29:41] Vérification du statut de l'API...
Statut de l'API:
{
  "config_path": "/home/didier/Documents/OpenClassrooms/Projet7/Livrables/DracheDidier_1_API_032025/air-paradis-sentiment-analysis/model_config.json",
  "is_custom_config": false,
  "model_info": {
    "has_predict": true,
    "has_predict_proba": false,
    "type": "Sequential"
  },
  "model_loaded": true,
  "model_path": "/home/didier/Documents/OpenClassrooms/Projet7/Livrables/DracheDidier_1_API_032025/air-paradis-sentiment-analysis/models/deeplearning/cnn_(embeddings_entrainables).keras",
  "model_type": "deeplearning",
  "status": "API operationnelle",
  "tokenizer_loaded": true
}
Temps de réponse: 0.0032 secondes
--------------------------------------------------
Test 1: I absolutely love this airline! Best flight ever!
Sentiment: Positif
Probabilités: Positif=0.9928, Negatif=0.0072
Temps de prédiction: 0.0381 secondes
--------------------------------------------------
Test 2: This is the worst airline experience I've ever had.
Sentiment: Negatif
Probabilités: Positif=0.0677, Negatif=0.9323
Temps de prédiction: 0.0401 secondes
--------------------------------------------------
Test 3: The flight was delayed by 2 hours and no compensation was offered.
Sentiment: Negatif
Probabilités: Positif=0.0252, Negatif=0.9748
Temps de prédiction: 0.0378 secondes
--------------------------------------------------
Test 4: Air Paradis has the best customer service I've experienced.
Sentiment: Positif
Probabilités: Positif=0.8798, Negatif=0.1202
Temps de prédiction: 0.0373 secondes
--------------------------------------------------
Test 5: Not sure how I feel about this flight, it's just okay I guess.
Sentiment: Positif
Probabilités: Positif=0.6312, Negatif=0.3688
Temps de prédiction: 0.0360 secondes
--------------------------------------------------

Statistiques de performance:
Modèle testé: deeplearning (Sequential)
Chemin du modèle: /home/didier/Documents/OpenClassrooms/Projet7/Livrables/DracheDidier_1_API_032025/air-paradis-sentiment-analysis/models/deeplearning/cnn_(embeddings_entrainables).keras
Temps moyen de prédiction: 0.0379 secondes
Temps médian de prédiction: 0.0378 secondes
Temps minimum de prédiction: 0.0360 secondes
Temps maximum de prédiction: 0.0401 secondes
Écart type: 0.0015 secondes si plus de 2 valeurs
```

### Exemple d'appel API avec curl

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love flying with Air Paradis!"}'
{"cleaned_text":"i absolutely love flying with air paradis","confidence":0.9939167499542236,"input_text":"I absolutely love flying with Air Paradis!","label":"Positif","model_type":"deeplearning","prediction":1,"probabilities":{"Negatif":0.006083250045776367,"Positif":0.9939167499542236},"timestamp":"2025-05-15T14:34:02.058239"}
```

### Test avec l'API déployée

```bash
python test_api.py https://air-paradis-sentiment-api-cnn-embed2.azurewebsites.net
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

Le projet global explore plusieurs modèles, mais cette API utilise spécifiquement le modèle CNN avec embeddings entraînables:

| Modèle | Type | Taille | Performance |
|--------|------|--------|-------------|
| CNN (embeddings entraînables) | Deep Learning | ~30 Mo | F1 Score: 0.XX |

Les autres modèles étudiés dans le cadre du projet global incluent:
- Modèles classiques (Régression Logistique, SVM, etc.)
- Autres architectures deep learning (LSTM, CNN avec GloVe)
- Modèles Transformer (BERT, DistilBERT)

## Endpoints de l'API

- `GET /`: Page d'accueil avec informations sur l'API
- `GET /status`: État de l'API et informations sur le modèle chargé
- `POST /predict`: Analyse le sentiment d'un texte fourni
  - Body: `{"text": "Votre texte à analyser"}`
  - Réponse: Sentiment, confiance et probabilités
- `POST /feedback`: Soumet un feedback sur une prédiction pour amélioration continue
  - Body: `{"text": "texte", "predicted_sentiment": "...", "actual_sentiment": "..."}`
- `GET/POST /config`: Obtenir ou modifier la configuration du modèle
- `GET /ping`: Simple vérification que l'API est en fonctionnement

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
