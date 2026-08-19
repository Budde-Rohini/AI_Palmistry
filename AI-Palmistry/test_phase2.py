import io
import os
import unittest
import numpy as np
import cv2
from PIL import Image
import sys

# Ensure project directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from database.db import get_reading

class TestPhase2Upload(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        with self.client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'testuser'


    def create_dummy_image_stream(self, width=800, height=800, fmt='JPEG'):
        """Helper to generate in-memory JPEG/PNG image bytes."""
        img = Image.new('RGB', (width, height), color=(200, 160, 140))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=fmt)
        img_byte_arr.seek(0)
        return img_byte_arr

    def test_valid_image_upload(self):
        img_stream = self.create_dummy_image_stream(800, 800, 'JPEG')
        response = self.client.post('/upload', data={
            'palm_image': (img_stream, 'test_palm.jpg')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data['success'])
        self.assertIn('reading_id', json_data)

        # Check DB record creation
        reading = get_reading(json_data['reading_id'])
        self.assertIsNotNone(reading)
        self.assertIn('_original.jpg', reading['image_path'])

    def test_invalid_extension(self):
        txt_stream = io.BytesIO(b"Not an image file content.")
        response = self.client.post('/upload', data={
            'palm_image': (txt_stream, 'document.pdf')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data['success'])
        self.assertIn('Unsupported file format', json_data['error'])

    def test_low_resolution_image(self):
        img_stream = self.create_dummy_image_stream(150, 150, 'JPEG')
        response = self.client.post('/upload', data={
            'palm_image': (img_stream, 'tiny_palm.jpg')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data['success'])
        self.assertIn('too low', json_data['error'])

    def test_corrupted_image_file(self):
        corrupt_stream = io.BytesIO(b"\xFF\xD8\xFF\xE0FakeHeaderNotAnImageBodyData123456")
        response = self.client.post('/upload', data={
            'palm_image': (corrupt_stream, 'corrupt.jpg')
        }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertFalse(json_data['success'])
        self.assertIn('Corrupted or invalid image file', json_data['error'])

if __name__ == '__main__':
    unittest.main()
