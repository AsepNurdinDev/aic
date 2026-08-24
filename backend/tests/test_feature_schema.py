import unittest
import os
import json
from fusion.feature_schema import DrowsinessOutput, StateFarmOutput, RoadOutput

class TestFeatureSchema(unittest.TestCase):
    def test_drowsiness_schema(self):
        obj = DrowsinessOutput(drowsiness_available=1, drowsy_prob=0.85)
        self.assertEqual(obj.drowsiness_available, 1)
        self.assertEqual(obj.drowsy_prob, 0.85)
        self.assertEqual(obj.not_drowsy_prob, 0.0) # default
        
    def test_statefarm_schema(self):
        obj = StateFarmOutput(statefarm_available=1, sf_drinking_prob=0.9, sf_predicted_class="drinking")
        self.assertEqual(obj.statefarm_available, 1)
        self.assertEqual(obj.sf_drinking_prob, 0.9)
        self.assertEqual(obj.sf_predicted_class, "drinking")
        
    def test_road_schema(self):
        obj = RoadOutput(road_available=1, road_object_count=5)
        self.assertEqual(obj.road_available, 1)
        self.assertEqual(obj.road_object_count, 5)

    def test_schema_json_exists_and_valid(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(base_dir, "configs", "decision_feature_schema.json")
        self.assertTrue(os.path.exists(schema_path))
        with open(schema_path, "r") as f:
            data = json.load(f)
            self.assertIn("features", data)
            self.assertTrue(len(data["features"]) > 0)
