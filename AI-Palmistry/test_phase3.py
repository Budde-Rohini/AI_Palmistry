import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from palm_analysis.hand_detection import detect_hand
from palm_analysis.palm_detection import extract_palm_roi

class TestPhase3HandDetection(unittest.TestCase):

    def test_opencv_fallback_on_blank_image(self):
        # 500x500 synthetic image with skin tone box in center
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        # Add skin-tone region in center
        img[100:400, 100:400] = [140, 160, 220]  # BGR skin color approximation

        hand_data = detect_hand(img)
        self.assertIsNotNone(hand_data)
        self.assertTrue('detected' in hand_data)
        self.assertIn(hand_data['detection_method'], ['mediapipe', 'opencv_fallback', 'none'])
        self.assertIn('landmark_overlay', hand_data)
        self.assertEqual(hand_data['landmark_overlay'].shape, img.shape)

    def test_palm_roi_extraction(self):
        img = np.ones((600, 600, 3), dtype=np.uint8) * 128
        hand_data = {
            'detected': True,
            'detection_method': 'test',
            'landmarks': [(300, 500)] + [(250 + i*20, 200) for i in range(20)],
            'bbox': (150, 150, 450, 550)
        }

        palm_roi, roi_bbox = extract_palm_roi(img, hand_data)
        self.assertIsNotNone(palm_roi)
        self.assertGreater(palm_roi.shape[0], 0)
        self.assertGreater(palm_roi.shape[1], 0)
        self.assertEqual(len(roi_bbox), 4)

    def test_fallback_on_black_image(self):
        black_img = np.zeros((400, 400, 3), dtype=np.uint8)
        hand_data = detect_hand(black_img)
        self.assertIsNotNone(hand_data)
        self.assertEqual(hand_data['detected'], False)

        palm_roi, roi_bbox = extract_palm_roi(black_img, hand_data)
        self.assertEqual(palm_roi.shape[0] > 0, True)

if __name__ == '__main__':
    unittest.main()
