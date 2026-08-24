import json
import os
from typing import Dict, Any, List

class DecisionInput:
    """
    Contract for what the future Decision Model will receive.
    """
    def __init__(self, 
                 drowsiness_features: Dict[str, Any],
                 statefarm_features: Dict[str, Any],
                 road_features: Dict[str, Any],
                 temporal_features: Dict[str, Any],
                 availability: Dict[str, int],
                 timestamp: float):
        self.drowsiness = drowsiness_features
        self.statefarm = statefarm_features
        self.road = road_features
        self.temporal = temporal_features
        self.availability = availability
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drowsiness": self.drowsiness,
            "statefarm": self.statefarm,
            "road": self.road,
            "temporal": self.temporal,
            "availability": self.availability,
            "timestamp": self.timestamp
        }
    
    def to_flattened_vector(self, schema_path: str) -> List[float]:
        """
        Produces a deterministic flattened numerical vector based on the defined schema.
        String fields (like context, state, etc.) are excluded or should be mapped before this.
        """
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Feature schema not found at {schema_path}")
            
        with open(schema_path, "r") as f:
            schema = json.load(f)
            
        vector = []
        all_features = {**self.drowsiness, **self.statefarm, **self.road, **self.temporal, **self.availability}
        
        for feature_def in schema["features"]:
            if feature_def["type"] in ["float", "int"]:
                val = all_features.get(feature_def["name"], 0.0)
                vector.append(float(val))
                
        return vector
