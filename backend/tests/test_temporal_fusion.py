import unittest
from fusion.feature_schema import DrowsinessOutput, StateFarmOutput, RoadOutput
from fusion.temporal_fusion import TemporalFusion

class TestTemporalFusion(unittest.TestCase):
    def test_temporal_window(self):
        fusion = TemporalFusion(window_size=3)
        
        d1 = DrowsinessOutput(drowsiness_available=1, drowsy_prob=0.8, is_drowsy=1)
        s1 = StateFarmOutput(statefarm_available=1, sf_confidence=0.9)
        r1 = RoadOutput(road_available=1, road_score=50, road_approaching_count=1)
        
        f1 = fusion.update(d1, s1, r1)
        self.assertEqual(f1["temporal_mean_drowsy_prob"], 0.8)
        self.assertEqual(f1["temporal_drowsy_duration"], 0.2)
        
        d2 = DrowsinessOutput(drowsiness_available=1, drowsy_prob=0.4, is_drowsy=0)
        f2 = fusion.update(d2, s1, r1)
        self.assertAlmostEqual(f2["temporal_mean_drowsy_prob"], 0.6)
        self.assertAlmostEqual(f2["temporal_drowsy_duration"], 0.2)
        self.assertAlmostEqual(f2["temporal_approaching_persistence"], 1.0)
        
        # Test deterministic features over window size
        d3 = DrowsinessOutput(drowsiness_available=1, drowsy_prob=0.0, is_drowsy=0)
        fusion.update(d3, s1, r1)
        
        d4 = DrowsinessOutput(drowsiness_available=1, drowsy_prob=0.0, is_drowsy=0)
        f4 = fusion.update(d4, s1, r1)
        
        # The window size is 3, so d1 (0.8) should be pushed out.
        # Window: d2 (0.4), d3 (0.0), d4 (0.0) -> mean = 0.4 / 3 = 0.1333...
        self.assertAlmostEqual(f4["temporal_mean_drowsy_prob"], 0.4 / 3.0)
