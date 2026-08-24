from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class DrowsinessOutput:
    drowsiness_available: int = 0
    drowsy_prob: float = 0.0
    not_drowsy_prob: float = 0.0
    drowsiness_state: str = "UNKNOWN"
    is_drowsy: int = 0
    alarm_active: int = 0

@dataclass
class StateFarmOutput:
    statefarm_available: int = 0
    sf_drinking_prob: float = 0.0
    sf_normal_prob: float = 0.0
    sf_phone_prob: float = 0.0
    sf_radio_prob: float = 0.0
    sf_reaching_prob: float = 0.0
    sf_passenger_prob: float = 0.0
    sf_texting_prob: float = 0.0
    sf_predicted_class: str = "UNKNOWN"
    sf_confidence: float = 0.0

@dataclass
class RoadOutput:
    road_available: int = 0
    road_object_count: int = 0
    road_car_count: int = 0
    road_truck_count: int = 0
    road_bus_count: int = 0
    road_two_wheeler_count: int = 0
    road_person_count: int = 0
    road_max_confidence: float = 0.0
    road_nearest_object_confidence: float = 0.0
    road_drivable_fraction: float = 0.0
    road_lane_fraction: float = 0.0
    road_drivable_overlap: float = 0.0
    road_lane_distance: float = 0.0
    road_score: float = 0.0
    road_approaching_count: int = 0
    road_growth: float = 0.0
    road_slope: float = 0.0
    road_trend_consistency: float = 0.0
    road_bottom_y_growth: float = 0.0
    road_motion: float = 0.0
    road_context: str = "UNKNOWN"
    road_proximity: float = 0.0
    road_lane_relation: str = "UNKNOWN"

@dataclass
class SensorAvailability:
    drowsiness: int = 0
    statefarm: int = 0
    road: int = 0
