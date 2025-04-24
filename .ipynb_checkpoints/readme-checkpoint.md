# contenu du répertoire models

❯ find ./models -name "*.*" -exec ls -la {} \;
-rw-r--r-- 1 didier didier 58976791 17 avril 22:50 ./models/classical/random_forest.pkl
-rw-r--r-- 1 didier didier 53672867 17 avril 22:47 ./models/classical/naive_bayes.pkl
-rw-r--r-- 1 didier didier 52472876 24 avril 09:51 ./models/classical/svm_lineaire.pkl
-rw-r--r-- 1 didier didier 52472962 24 avril 09:51 ./models/classical/regression_logistique.pkl
-rw-r--r-- 1 didier didier 267951808 18 avril 08:20 ./models/bert/distilbert_base/tf_model.h5
-rw-r--r-- 1 didier didier 538 18 avril 08:20 ./models/bert/distilbert_base/config.json
-rw-r--r-- 1 didier didier 438223032 18 avril 07:04 ./models/bert/bert_base/tf_model.h5
-rw-r--r-- 1 didier didier 651 18 avril 07:04 ./models/bert/bert_base/config.json
-rw-r--r-- 1 didier didier 16653358 17 avril 22:51 ./models/deeplearning/tokenizer.pkl
-rw-r--r-- 1 didier didier 20549771 17 avril 23:24 './models/deeplearning/cnn_(glove).keras'
-rw-r--r-- 1 didier didier 40000128 16 avril 22:21 ./models/deeplearning/random_100d_embedding.npy
-rw-r--r-- 1 didier didier 60448593 18 avril 02:10 './models/deeplearning/lstm_(embeddings_entrainables).keras'
-rw-r--r-- 1 didier didier 30358407 17 avril 23:08 './models/deeplearning/cnn_(embeddings_entrainables).keras'
-rw-r--r-- 1 didier didier 21407038 18 avril 04:47 './models/deeplearning/lstm_(glove).keras'

# environnements

python3.10 -m venv venv_api_classical
source venv_api_classical/bin/activate
pip install -r requirements_classical.txt
python api.py model_config_naive_bayes.json
python api.py model_config_random_forest.json
python api.py model_config_reglog.json
python api.py model_config_svm_lineaire.json

python3.10 -m venv venv_api_deeplearning
source venv_api_deeplearning/bin/activate
pip install -r requirements_deeplearning.txt
python api.py model_config_cnn_embeddings.json
python api.py model_config_cnn_glove.json
python api.py model_config_lstm_embeddings.json
python api.py model_config_lstm_glove.json


python3.10 -m venv venv_api_bert
source venv_api_bert/bin/activate
pip install -r requirements_bert.txt
python api.py model_config_bert.json
python api.py model_config_distilbert.json

# test

❯ python test_api.py
[13:08:34] Vérification du statut de l'API...
Statut de l'API:
{
  "config_path": "model_config_distilbert.json",
  "is_custom_config": true,
  "model_info": {
    "has_predict": true,
    "has_predict_proba": false,
    "type": "TFDistilBertForSequenceClassification"
  },
  "model_loaded": true,
  "model_path": "/home/didier/Documents/OpenClassrooms/Projet7/sentiment_api_test4/models/bert/distilbert_base",
  "model_type": "transformer",
  "status": "API operationnelle",
  "tokenizer_loaded": true
}
Temps de réponse: 0.0038 secondes
--------------------------------------------------
Test 1: I absolutely love this airline! Best flight ever!
Sentiment: Positif
Probabilités: Positif=0.9957, Negatif=0.0043
Temps de prédiction: 0.2721 secondes
--------------------------------------------------
Test 2: This is the worst airline experience I've ever had.
Sentiment: Negatif
Probabilités: Positif=0.0040, Negatif=0.9960
Temps de prédiction: 0.2648 secondes
--------------------------------------------------
Test 3: The flight was delayed by 2 hours and no compensation was offered.
Sentiment: Negatif
Probabilités: Positif=0.0074, Negatif=0.9926
Temps de prédiction: 0.3292 secondes
--------------------------------------------------
Test 4: Air Paradis has the best customer service I've experienced.
Sentiment: Positif
Probabilités: Positif=0.9875, Negatif=0.0125
Temps de prédiction: 0.2469 secondes
--------------------------------------------------
Test 5: Not sure how I feel about this flight, it's just okay I guess.
Sentiment: Negatif
Probabilités: Positif=0.1370, Negatif=0.8630
Temps de prédiction: 0.3087 secondes
--------------------------------------------------

Statistiques de performance:
Modèle testé: transformer (TFDistilBertForSequenceClassification)
Chemin du modèle: /home/didier/Documents/OpenClassrooms/Projet7/sentiment_api_test4/models/bert/distilbert_base
Temps moyen de prédiction: 0.2844 secondes
Temps médian de prédiction: 0.2721 secondes
Temps minimum de prédiction: 0.2469 secondes
Temps maximum de prédiction: 0.3292 secondes
Écart type: 0.0337 secondes si plus de 2 valeurs


