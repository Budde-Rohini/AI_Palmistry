import io
import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from database.db import get_parsed_reading

class TestPhase8And9Integration(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'testuser'


    def create_dummy_image_stream(self, width=600, height=600):
        img = Image.new('RGB', (width, height), color=(220, 180, 160))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr

    def test_full_pipeline_flow(self):
        # 1. Upload
        img_stream = self.create_dummy_image_stream(600, 600)
        up_res = self.client.post('/upload', data={
            'palm_image': (img_stream, 'test_pipeline_palm.jpg')
        }, content_type='multipart/form-data')

        self.assertEqual(up_res.status_code, 200)
        up_json = up_res.get_json()
        self.assertTrue(up_json['success'])
        reading_id = up_json['reading_id']

        # 2. Analysis page GET
        an_res = self.client.get(f'/analysis/{reading_id}')
        self.assertEqual(an_res.status_code, 200)

        # 3. Trigger Analysis API POST
        api_res = self.client.post(f'/api/analyze/{reading_id}')
        self.assertEqual(api_res.status_code, 200)
        api_json = api_res.get_json()
        self.assertTrue(api_json['success'])

        # 4. Result page GET
        res_page = self.client.get(f'/result/{reading_id}')
        self.assertEqual(res_page.status_code, 200)
        self.assertIn(b"Your Structured Palm Analysis", res_page.data)

        # 5. Reading JSON API GET
        reading_api = self.client.get(f'/api/reading/{reading_id}')
        self.assertEqual(reading_api.status_code, 200)
        read_json = reading_api.get_json()
        self.assertTrue(read_json['success'])
        self.assertIn('interpretation', read_json['reading'])

if __name__ == '__main__':
    unittest.main()
