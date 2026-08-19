import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from database.db import init_db, create_reading, get_reading, get_parsed_reading, update_reading_analysis, list_readings

class TestPhase7Database(unittest.TestCase):

    def setUp(self):
        init_db()

    def test_database_crud(self):
        import uuid
        reading_id = f"test_uuid_{uuid.uuid4()}"
        create_reading(reading_id, "uploads/test_orig.jpg", "processed/test_proc.jpg")


        # Fetch raw reading
        reading = get_reading(reading_id)
        self.assertIsNotNone(reading)
        self.assertEqual(reading['id'], reading_id)
        self.assertEqual(reading['image_path'], "uploads/test_orig.jpg")

        # Update reading with analysis
        palm_feats = {"aspect_ratio": 0.95}
        line_feats = {"heart_line": {"detected": True}}
        interp = {"summary": "Test summary"}

        update_reading_analysis(reading_id, "results/test_overlay.jpg", "Right", palm_feats, line_feats, interp)

        # Fetch parsed reading
        parsed = get_parsed_reading(reading_id)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['overlay_image_path'], "results/test_overlay.jpg")
        self.assertEqual(parsed['hand_type'], "Right")
        self.assertIsInstance(parsed['palm_features'], dict)
        self.assertEqual(parsed['palm_features']['aspect_ratio'], 0.95)

        # List readings
        readings_list = list_readings(10)
        self.assertGreater(len(readings_list), 0)

if __name__ == '__main__':
    unittest.main()
