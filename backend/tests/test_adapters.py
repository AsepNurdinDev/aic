import unittest
from adapters.drowsiness_adapter import DrowsinessAdapter
from adapters.statefarm_adapter import StateFarmAdapter
from adapters.road_adapter import RoadAdapter

class TestAdapters(unittest.TestCase):
    def test_drowsiness_adapter(self):
        raw_v5 = {
            "status": "DROWSY",
            "scenario": "MICROSLEEP",
            "risk_level": "CRITICAL",
            "risk_score": 100,
            "ear": 0.2,
            "mar": 0.1,
            "alarm_active": True,
            "raw_prob": 0.95
        }
        out = DrowsinessAdapter.adapt(raw_v5)
        self.assertEqual(out.drowsiness_available, 1)
        self.assertEqual(out.drowsy_prob, 0.95)
        self.assertEqual(out.is_drowsy, 1)
        self.assertEqual(out.drowsiness_state, "MICROSLEEP")
        
        # Test missing sensor
        out_missing = DrowsinessAdapter.adapt({})
        self.assertEqual(out_missing.drowsiness_available, 0)
        
    def test_statefarm_adapter(self):
        probs = [0.05, 0.8, 0.05, 0.05, 0.02, 0.02, 0.01]  # normal is highest
        out = StateFarmAdapter.adapt(probs)
        self.assertEqual(out.statefarm_available, 1)
        self.assertEqual(out.sf_normal_prob, 0.8)
        self.assertEqual(out.sf_predicted_class, "normal")
        self.assertEqual(out.sf_confidence, 0.8)
        
        # Probabilities should approximate to what we set
        self.assertAlmostEqual(out.sf_drinking_prob + out.sf_normal_prob + out.sf_phone_prob + 
                               out.sf_radio_prob + out.sf_reaching_prob + out.sf_passenger_prob + 
                               out.sf_texting_prob, 1.0)
        
    def test_road_adapter(self):
        raw_road = {
            "object_count": 3,
            "car_count": 2,
            "max_confidence": 0.9,
            "drivable_fraction": 0.5,
            "context": "highway"
        }
        out = RoadAdapter.adapt(raw_road)
        self.assertEqual(out.road_available, 1)
        self.assertEqual(out.road_object_count, 3)
        self.assertEqual(out.road_car_count, 2)
        self.assertEqual(out.road_max_confidence, 0.9)
        self.assertEqual(out.road_drivable_fraction, 0.5)
        self.assertEqual(out.road_context, "highway")
