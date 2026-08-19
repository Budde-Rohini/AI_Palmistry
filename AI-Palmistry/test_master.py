import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_phase2 import TestPhase2Upload
from test_phase3 import TestPhase3HandDetection
from test_phase4 import TestPhase4LineDetection
from test_phase5 import TestPhase5FeatureExtraction
from test_phase6 import TestPhase6Interpretation
from test_phase7 import TestPhase7Database
from test_phase8_9 import TestPhase8And9Integration
from test_phase10 import TestPhase10Chatbot
from test_auth import TestAuthentication

def suite():
    test_suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    test_suite.addTest(loader.loadTestsFromTestCase(TestAuthentication))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase2Upload))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase3HandDetection))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase4LineDetection))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase5FeatureExtraction))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase6Interpretation))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase7Database))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase8And9Integration))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPhase10Chatbot))

    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    if not result.wasSuccessful():
        sys.exit(1)
