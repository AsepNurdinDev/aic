from collections import deque
import statistics
from typing import Dict, Any, List

class TemporalFusion:
    """
    Builds temporal feature history before Decision Model.
    Maintains a configurable history window (e.g., 2 seconds @ 5 Hz = 10 frames).
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.drowsy_prob_history = deque(maxlen=window_size)
        self.drowsy_state_history = deque(maxlen=window_size)
        self.sf_drinking_prob_history = deque(maxlen=window_size)
        self.sf_phone_prob_history = deque(maxlen=window_size)
        self.sf_texting_prob_history = deque(maxlen=window_size)
        self.sf_confidence_history = deque(maxlen=window_size)
        self.road_score_history = deque(maxlen=window_size)
        self.approaching_count_history = deque(maxlen=window_size)

    def update(self, drowsiness: Any, statefarm: Any, road: Any) -> Dict[str, float]:
        """
        Takes standardized output objects, updates history, and returns temporal features.
        """
        # Update Drowsiness History
        if drowsiness.drowsiness_available:
            self.drowsy_prob_history.append(drowsiness.drowsy_prob)
            self.drowsy_state_history.append(drowsiness.is_drowsy)
            
        # Update StateFarm History
        if statefarm.statefarm_available:
            self.sf_drinking_prob_history.append(statefarm.sf_drinking_prob)
            self.sf_phone_prob_history.append(statefarm.sf_phone_prob)
            self.sf_texting_prob_history.append(statefarm.sf_texting_prob)
            self.sf_confidence_history.append(statefarm.sf_confidence)
            
        # Update Road History
        if road.road_available:
            self.road_score_history.append(road.road_score)
            self.approaching_count_history.append(road.road_approaching_count)
            
        return self._calculate_features()

    def _calculate_features(self) -> Dict[str, float]:
        features = {}
        
        # Drowsiness temporal features
        if self.drowsy_prob_history:
            features["temporal_mean_drowsy_prob"] = statistics.mean(self.drowsy_prob_history)
            features["temporal_max_drowsy_prob"] = max(self.drowsy_prob_history)
        else:
            features["temporal_mean_drowsy_prob"] = 0.0
            features["temporal_max_drowsy_prob"] = 0.0
            
        if self.drowsy_state_history:
            # Approximation of duration given 5 Hz
            features["temporal_drowsy_duration"] = sum(self.drowsy_state_history) * 0.2
            # V5 doesn't provide yawning in binary flag, so we default this to 0 for now
            features["temporal_yawning_duration"] = 0.0
        else:
            features["temporal_drowsy_duration"] = 0.0
            features["temporal_yawning_duration"] = 0.0
            
        # StateFarm temporal features
        if self.sf_drinking_prob_history:
            features["temporal_mean_sf_drinking_prob"] = statistics.mean(self.sf_drinking_prob_history)
            features["temporal_mean_sf_phone_prob"] = statistics.mean(self.sf_phone_prob_history)
            features["temporal_mean_sf_texting_prob"] = statistics.mean(self.sf_texting_prob_history)
            features["temporal_max_sf_confidence"] = max(self.sf_confidence_history)
        else:
            features["temporal_mean_sf_drinking_prob"] = 0.0
            features["temporal_mean_sf_phone_prob"] = 0.0
            features["temporal_mean_sf_texting_prob"] = 0.0
            features["temporal_max_sf_confidence"] = 0.0
            
        # Road temporal features
        if self.road_score_history:
            features["temporal_mean_road_score"] = statistics.mean(self.road_score_history)
            features["temporal_max_road_score"] = max(self.road_score_history)
        else:
            features["temporal_mean_road_score"] = 0.0
            features["temporal_max_road_score"] = 0.0
            
        if self.approaching_count_history:
            # Persistence: ratio of frames where approaching objects > 0
            approaching_frames = sum(1 for c in self.approaching_count_history if c > 0)
            features["temporal_approaching_persistence"] = approaching_frames / len(self.approaching_count_history)
        else:
            features["temporal_approaching_persistence"] = 0.0
            
        return features
