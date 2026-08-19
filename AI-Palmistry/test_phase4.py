import os
import sys
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from palm_analysis.line_detection import detect_palm_lines

class TestPhase4LineDetection(unittest.TestCase):

    def test_line_detection_on_synthetic_palm(self):
        # Create 400x400 synthetic palm image with dark lines drawn
        palm_roi = np.ones((400, 400, 3), dtype=np.uint8) * 180  # Skin-like background
        
        # Draw synthetic creases (dark gray lines)
        cv2.line(palm_roi, (50, 100), (350, 120), (50, 50, 50), 3)  # Heart line
        cv2.line(palm_roi, (60, 180), (340, 220), (50, 50, 50), 3)  # Head line
        cv2.ellipse(palm_roi, (120, 250), (80, 120), 0, 0, 180, (50, 50, 50), 3) # Life line
        cv2.line(palm_roi, (200, 350), (200, 150), (50, 50, 50), 3) # Fate line

        res = detect_palm_lines(palm_roi, hand_type="Right")

        self.assertIsNotNone(res)
        self.assertIn('lines', res)
        self.assertIn('overlay_image', res)
        self.assertEqual(res['overlay_image'].shape, palm_roi.shape)

        lines = res['lines']
        self.assertIn('heart_line', lines)
        self.assertIn('head_line', lines)
        self.assertIn('life_line', lines)
        self.assertIn('fate_line', lines)

    def test_empty_roi_handling(self):
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        res = detect_palm_lines(empty_img)
        self.assertIsNotNone(res)
        self.assertIn('lines', res)

if __name__ == '__main__':
    unittest.main()
