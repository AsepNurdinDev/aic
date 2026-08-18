"""
backend/tests/test_model.py

Test untuk memverifikasi inisialisasi modul AI di backend.
"""

import os
import sys
import unittest
import numpy as np

# Tambahkan backend path ke sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.driver_analyzer import DriverAnalyzer
from src.config import MODEL_PATH, CONFIG_PATH, MEDIAPIPE_MODEL_PATH


class TestBackendAI(unittest.TestCase):

    def test_paths_exist(self):
        self.assertTrue(os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}")
        self.assertTrue(os.path.exists(CONFIG_PATH), f"Config path not found: {CONFIG_PATH}")
        self.assertTrue(os.path.exists(MEDIAPIPE_MODEL_PATH), f"MediaPipe model path not found: {MEDIAPIPE_MODEL_PATH}")

    def test_analyzer_initialization_and_dummy_frame(self):
        analyzer = DriverAnalyzer(
            checkpoint_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            mediapipe_model_path=MEDIAPIPE_MODEL_PATH,
            device="cpu"
        )
        self.assertIsNotNone(analyzer)

        # Process dummy black frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = analyzer.process_frame(dummy_frame)

        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertIn("risk_level", result)
        self.assertIn("risk_score", result)
        self.assertIn("ear", result)
        self.assertIn("mar", result)
        self.assertEqual(result["face_detected"], False)
        self.assertEqual(result["scenario"], "NO_FACE")
        self.assertEqual(result["risk_level"], "POTENTIAL")
        print("\n[SUCCESS] Backend AI Analyzer initialized & verified successfully with dummy frame!")


if __name__ == "__main__":
    unittest.main()
