from typing import Dict, Any, List
from fusion.feature_schema import RoadOutput

class RoadAdapter:
    """
    Adapter to convert Road Object + Geometry outputs into standardized RoadOutput.
    Assumes input is already at the frame-level or aggregates object-level into frame-level.
    """
    
    @staticmethod
    def adapt(road_data: Dict[str, Any]) -> RoadOutput:
        if not road_data:
            return RoadOutput(road_available=0)
            
        return RoadOutput(
            road_available=1,
            road_object_count=int(road_data.get("object_count", 0)),
            road_car_count=int(road_data.get("car_count", 0)),
            road_truck_count=int(road_data.get("truck_count", 0)),
            road_bus_count=int(road_data.get("bus_count", 0)),
            road_two_wheeler_count=int(road_data.get("two_wheeler_count", 0)),
            road_person_count=int(road_data.get("person_count", 0)),
            road_max_confidence=float(road_data.get("max_confidence", 0.0)),
            road_nearest_object_confidence=float(road_data.get("nearest_object_confidence", 0.0)),
            road_drivable_fraction=float(road_data.get("drivable_fraction", 0.0)),
            road_lane_fraction=float(road_data.get("lane_fraction", 0.0)),
            road_drivable_overlap=float(road_data.get("drivable_overlap", 0.0)),
            road_lane_distance=float(road_data.get("lane_distance", 0.0)),
            road_score=float(road_data.get("score", 0.0)),
            road_approaching_count=int(road_data.get("approaching_count", 0)),
            road_growth=float(road_data.get("growth", 0.0)),
            road_slope=float(road_data.get("slope", 0.0)),
            road_trend_consistency=float(road_data.get("trend_consistency", 0.0)),
            road_bottom_y_growth=float(road_data.get("bottom_y_growth", 0.0)),
            road_motion=float(road_data.get("motion", 0.0)),
            road_context=str(road_data.get("context", "UNKNOWN")),
            road_proximity=float(road_data.get("proximity", 0.0)),
            road_lane_relation=str(road_data.get("lane_relation", "UNKNOWN"))
        )
