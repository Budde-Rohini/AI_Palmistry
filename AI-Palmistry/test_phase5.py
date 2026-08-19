import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from palm_analysis.feature_extraction import extract_features

class TestPhase5FeatureExtraction(unittest.TestCase):

    def test_feature_vector_structure(self):
        hand_data = {
            'hand_type': 'Right',
            'detection_method': 'mediapipe',
            'detected': True,
            'palm_width': 220.0,
            'palm_height': 240.0,
            'aspect_ratio': 0.917,
            'finger_lengths': {'index': 110.0, 'middle': 130.0, 'ring': 120.0, 'pinky': 90.0, 'thumb': 80.0}
        }

        line_data = {
            'lines': {
                'heart_line': {'detected': True, 'length': 'Long', 'length_px': 210.0, 'curvature': 'Strong / Curved', 'depth': 'Clear', 'continuity': 'Continuous'},
                'head_line': {'detected': True, 'length': 'Medium', 'length_px': 180.0, 'curvature': 'Moderate', 'depth': 'Clear', 'continuity': 'Continuous'},
                'life_line': {'detected': True, 'length': 'Long', 'length_px': 260.0, 'curvature': 'Strong / Curved', 'depth': 'Clear', 'continuity': 'Continuous'},
                'fate_line': {'detected': False, 'length': 'Unavailable', 'curvature': 'Unavailable', 'depth': 'Unavailable', 'continuity': 'Unavailable'}
            }
        }

        features = extract_features(hand_data, line_data)

        self.assertIsNotNone(features)
        self.assertEqual(features['hand'], 'Right')
        self.assertEqual(features['palm']['aspect_ratio'], 0.917)
        self.assertEqual(features['heart_line']['detected'], True)
        self.assertEqual(features['heart_line']['length'], 'Long')
        self.assertEqual(features['fate_line']['detected'], False)
        self.assertEqual(features['fate_line']['length'], 'Unavailable')
        self.assertGreater(features['quality_score'], 50)

    def test_empty_hand_and_lines(self):
        features = extract_features({}, {})
        self.assertEqual(features['hand'], 'Unknown')
        self.assertEqual(features['life_line']['detected'], False)

if __name__ == '__main__':
    unittest.main()
