import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from palm_analysis.interpretation import generate_interpretation

class TestPhase6Interpretation(unittest.TestCase):

    def test_generate_interpretation(self):
        features = {
            "hand": "Right",
            "palm": {
                "aspect_ratio": 0.92,
                "element_shape": "Earth (Square)"
            },
            "heart_line": {"detected": True, "length": "Long", "curvature": "Strong / Curved"},
            "head_line": {"detected": True, "length": "Long", "curvature": "Moderate"},
            "life_line": {"detected": True, "length": "Long", "curvature": "Strong / Curved"},
            "fate_line": {"detected": False}
        }

        interp = generate_interpretation(features)

        self.assertIsNotNone(interp)
        self.assertIn('disclaimer', interp)
        self.assertIn('major_lines', interp)
        self.assertIn('life_domains', interp)
        self.assertIn('summary_report', interp)

        domains = interp['life_domains']
        self.assertIn('career', domains)
        self.assertIn('education', domains)
        self.assertIn('finance', domains)
        self.assertIn('relationships', domains)
        self.assertIn('personality', domains)
        self.assertIn('general_vitality', domains)

        # Verify disclaimers present
        self.assertIn('entertainment', interp['disclaimer'].lower())
        self.assertIn('not medical', domains['general_vitality']['traditional_insight'].lower())
        self.assertIn('financial advice', domains['finance']['traditional_insight'].lower())


    def test_missing_features(self):
        interp = generate_interpretation({})
        self.assertIsNotNone(interp)
        self.assertIn('disclaimer', interp)

if __name__ == '__main__':
    unittest.main()
