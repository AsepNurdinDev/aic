import time
from typing import Dict, Any, Optional

from adapters.drowsiness_adapter import DrowsinessAdapter
from adapters.statefarm_adapter import StateFarmAdapter
from adapters.road_adapter import RoadAdapter
from .temporal_fusion import TemporalFusion
from .decision_input import DecisionInput
import os
from dataclasses import asdict

class DecisionPipeline:
    """
    Demo Inference Pipeline that processes outputs from 3 models and prepares them 
    for the future Decision Model.
    """
    def __init__(self, schema_path: str = None):
        self.temporal_fusion = TemporalFusion(window_size=10) # 2 seconds @ 5Hz
        if schema_path is None:
            # Default to configs/decision_feature_schema.json relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.schema_path = os.path.join(base_dir, "configs", "decision_feature_schema.json")
        else:
            self.schema_path = schema_path

    def process(self, 
                drowsiness_raw: Optional[Dict[str, Any]] = None,
                statefarm_raw: Optional[list] = None,
                road_raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Takes raw outputs, standardizes them, computes temporal features, 
        and prepares the final feature vector.
        """
        # 1. Adapt raw outputs to standard schema
        drowsiness_out = DrowsinessAdapter.adapt(drowsiness_raw)
        statefarm_out = StateFarmAdapter.adapt(statefarm_raw)
        road_out = RoadAdapter.adapt(road_raw)
        
        # 2. Check sensor availability
        sensors_available = {
            "drowsiness": bool(drowsiness_out.drowsiness_available),
            "statefarm": bool(statefarm_out.statefarm_available),
            "road": bool(road_out.road_available)
        }
        
        availability_features = {
            "drowsiness": int(sensors_available["drowsiness"]),
            "statefarm": int(sensors_available["statefarm"]),
            "road": int(sensors_available["road"])
        }
        
        # 3. Update temporal fusion
        temporal_features = self.temporal_fusion.update(drowsiness_out, statefarm_out, road_out)
        
        # 4. Build Decision Input contract
        current_time = time.time()
        decision_input = DecisionInput(
            drowsiness_features=asdict(drowsiness_out),
            statefarm_features=asdict(statefarm_out),
            road_features=asdict(road_out),
            temporal_features=temporal_features,
            availability=availability_features,
            timestamp=current_time
        )
        
        # Flatten vector using schema
        try:
            flattened_vector = decision_input.to_flattened_vector(self.schema_path)
            ready = any(sensors_available.values()) # Ready if at least one sensor is active
            pipeline_status = "OK" if ready else "SENSOR_FAILURE"
        except Exception as e:
            flattened_vector = []
            ready = False
            pipeline_status = f"ERROR: {str(e)}"
            
        if not ready:
            pipeline_status = "SENSOR_FAILURE"
            
        return {
            "pipeline_status": pipeline_status,
            "ready_for_decision_model": ready,
            "sensor_status": sensors_available,
            "feature_vector": flattened_vector,
            "raw_features": decision_input.to_dict(),
            "timestamp": current_time
        }
