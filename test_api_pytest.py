import json
import pytest
from api import app, clean_text

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def test_clean_text():
    # Test de nettoyage de texte basique
    text = "Hello @user! Check out https://example.com #example"
    expected = "hello check out example"
    assert clean_text(text) == expected
    
    # Test avec texte vide
    assert clean_text("") == ""
    
    # Test avec input non-texte
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_predict_endpoint(client):
    # Test avec un tweet positif
    response = client.post('/predict', 
                         data=json.dumps({'text': 'I love this airline!'}),
                         content_type='application/json')
    data = json.loads(response.data)
    assert response.status_code == 200
    assert 'label' in data
    assert 'probabilities' in data
    
    # Test avec requête mal formée
    response = client.post('/predict', 
                         data=json.dumps({'wrong_field': 'text'}),
                         content_type='application/json')
    assert response.status_code == 400

def test_status_endpoint(client):
    response = client.get('/status')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert 'model_loaded' in data
    assert data['model_loaded'] is True  # Vérifie que le modèle est chargé

def test_root_endpoint(client):
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'name' in data
    assert 'version' in data
    assert 'endpoints' in data

def test_prediction_with_different_inputs(client):
    # Test avec un tweet clairement positif
    positive_text = "I absolutely love flying with Air Paradis! Best experience ever!"
    response = client.post('/predict', 
                        data=json.dumps({'text': positive_text}),
                        content_type='application/json')
    data = json.loads(response.data)
    assert data['label'] == "Positif"  # Doit prédire positif
    
    # Test avec un tweet clairement négatif
    negative_text = "Worst airline ever! Terrible service and always delayed."
    response = client.post('/predict', 
                        data=json.dumps({'text': negative_text}),
                        content_type='application/json')
    data = json.loads(response.data)
    assert data['label'] == "Negatif"  # Doit prédire négatif
    
    # Test avec un texte vide
    response = client.post('/predict', 
                        data=json.dumps({'text': ""}),
                        content_type='application/json')
    assert response.status_code == 400  # Devrait retourner une erreur 400