import io
import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app

class TestPhase10Chatbot(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'testuser'


    def create_dummy_reading(self):
        img = Image.new('RGB', (500, 500), color=(200, 160, 140))
        img_byte = io.BytesIO()
        img.save(img_byte, format='JPEG')
        img_byte.seek(0)

        up_res = self.client.post('/upload', data={'palm_image': (img_byte, 'palm_chat_test.jpg')}, content_type='multipart/form-data')
        reading_id = up_res.get_json()['reading_id']
        self.client.post(f'/api/analyze/{reading_id}')
        return reading_id

    def test_chatbot_grounded_qa(self):
        reading_id = self.create_dummy_reading()

        # Test Heart Line question
        res = self.client.post('/api/chat', json={
            'reading_id': reading_id,
            'question': 'What does my heart line mean?'
        })
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertTrue(json_data['success'])
        self.assertIn('Heart Line', json_data['answer'])

        # Test Future / Prediction attempt
        res_future = self.client.post('/api/chat', json={
            'reading_id': reading_id,
            'question': 'Can you predict my future wealth?'
        })
        self.assertEqual(res_future.status_code, 200)
        self.assertIn('does not make deterministic predictions', res_future.get_json()['answer'].lower())


    def test_chatbot_invalid_reading(self):
        res = self.client.post('/api/chat', json={
            'reading_id': 'non_existent_id',
            'question': 'Hello'
        })
        self.assertEqual(res.status_code, 404)

if __name__ == '__main__':
    unittest.main()
