import unittest
import json
from api import app, clean_text

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_status_endpoint(self):
        response = self.app.get('/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        
    def test_clean_text_function(self):
        text = "Hello @user #test http://example.com"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "hello test")

if __name__ == '__main__':
    unittest.main()

    