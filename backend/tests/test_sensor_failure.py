import unittest
from fusion.pipeline import DecisionPipeline

class TestSensorFailure(unittest.TestCase):
    def setUp(self):
        self.pipeline = DecisionPipeline()
        
    def test_missing_drowsiness(self):
        statefarm_raw = [0.1, 0.7, 0.1, 0.05, 0.05, 0.0, 0.0]
        road_raw = {"object_count": 2, "max_confidence": 0.8}
        
        result = self.pipeline.process(
            drowsiness_raw=None, 
            statefarm_raw=statefarm_raw, 
            road_raw=road_raw
        )
        
        self.assertTrue(result["ready_for_decision_model"])
        self.assertEqual(result["sensor_status"]["drowsiness"], False)
        self.assertEqual(result["sensor_status"]["statefarm"], True)
        self.assertEqual(result["sensor_status"]["road"], True)
        self.assertEqual(result["pipeline_status"], "OK")
        
    def test_missing_all(self):
        result = self.pipeline.process(None, None, None)
        
        self.assertFalse(result["ready_for_decision_model"])
        self.assertEqual(result["pipeline_status"], "SENSOR_FAILURE")
        self.assertEqual(result["sensor_status"]["drowsiness"], False)
        self.assertEqual(result["sensor_status"]["statefarm"], False)
        self.assertEqual(result["sensor_status"]["road"], False)
        
    def test_missing_statefarm_and_road(self):
        drowsiness_raw = {"status": "DROWSY", "raw_prob": 0.8}
        
        result = self.pipeline.process(drowsiness_raw=drowsiness_raw, statefarm_raw=None, road_raw=None)
        
        self.assertTrue(result["ready_for_decision_model"])
        self.assertEqual(result["sensor_status"]["drowsiness"], True)
        self.assertEqual(result["sensor_status"]["statefarm"], False)
        self.assertEqual(result["sensor_status"]["road"], False)
        self.assertEqual(result["pipeline_status"], "OK")
