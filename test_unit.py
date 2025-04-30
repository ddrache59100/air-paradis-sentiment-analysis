import unittest
import json
from api import app, clean_text, predict_with_model

class TestAPIFunctions(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_clean_text(self):
        # Test de nettoyage de texte basique
        text = "Hello @user! Check out https://example.com #example"
        expected = "hello check out example"  # Correspond au comportement réel
        self.assertEqual(clean_text(text), expected)
        
        # Test avec texte vide
        self.assertEqual(clean_text(""), "")
        
        # Test avec input non-texte
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")

    def test_predict_endpoint(self):
        # Test avec un tweet positif
        response = self.app.post('/predict', 
                               data=json.dumps({'text': 'I love this airline!'}),
                               content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('label', data)
        self.assertIn('probabilities', data)
        
        # Test avec requête mal formée
        response = self.app.post('/predict', 
                               data=json.dumps({'wrong_field': 'text'}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()