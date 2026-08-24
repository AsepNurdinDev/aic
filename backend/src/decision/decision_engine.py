import os
import json
import numpy as np
import xgboost as xgb
from collections import deque
from .thresholds import ThresholdManager

class DecisionEngine:
    def __init__(self, models_dir: str, schema_path: str):
        self.threshold_mgr = ThresholdManager(os.path.join(models_dir, "FINAL_champion_config.json"))
        
        # Hardcode exactly what the model expects, since the schema file is out of date
        self.feature_names = [
            'drowsy_drowsy_prob', 'drowsy_not_drowsy_prob', 'drowsy_alert_prob', 
            'drowsy_microsleep_prob', 'drowsy_yawning_prob', 'drowsy_prediction_confidence', 
            'sf_drinking_prob', 'sf_normal_prob', 'sf_phone_prob', 'sf_radio_prob', 
            'sf_reaching_prob', 'sf_passenger_prob', 'sf_texting_prob', 'sf_behavior_max_prob', 
            'sf_distraction_prob', 'road_object_count', 'road_car_count', 'road_truck_count', 
            'road_bus_count', 'road_two_wheeler_count', 'road_person_count', 'road_max_object_confidence', 
            'road_nearest_object_confidence', 'road_nearest_object_area_ratio', 'road_drivable_fraction', 
            'road_lane_fraction', 'road_road_relevance_score', 'drowsiness_available', 
            'statefarm_available', 'road_available'
        ]
        
        # Load XGBoost models
        self.models = {}
        events = [
            "drowsiness", "phone_use", "texting", "drinking",
            "radio_operation", "reaching_behind", "talking_passenger", "road_risk"
        ]
        
        for event in events:
            model_path = os.path.join(models_dir, f"FINAL_event_{event}.json")
            if os.path.exists(model_path):
                booster = xgb.Booster()
                booster.load_model(model_path)
                self.models[event] = booster
            else:
                print(f"Warning: Model for {event} not found at {model_path}")
                
        # Temporal buffer for calculating temporal features
        self.history = deque(maxlen=16)

    def compute_road_relevance(self, road_features: dict) -> float:
        obj_count = road_features.get("object_count", 0)
        area_ratio = road_features.get("nearest_object_area_ratio", 0.0)
        vehicles = (road_features.get("car_count", 0) + 
                    road_features.get("truck_count", 0) + 
                    road_features.get("bus_count", 0) + 
                    road_features.get("two_wheeler_count", 0))
                    
        object_density_score = np.clip(obj_count / 20.0, 0, 1)
        nearest_size_score = np.clip(area_ratio / 0.20, 0, 1)
        vehicle_density_score = np.clip(vehicles / 15.0, 0, 1)
        
        road_relevance = np.clip(
            0.40 * object_density_score + 
            0.35 * nearest_size_score + 
            0.25 * vehicle_density_score, 
            0, 1
        )
        return float(road_relevance)

    def process(self, timestamp: float, 
                drowsiness_res: dict, 
                statefarm_res: dict, 
                road_obj_res: dict, 
                road_geom_res: dict) -> dict:
        
        # Availability
        avail = {
            "drowsiness": drowsiness_res.get("drowsiness_available", 0),
            "statefarm": statefarm_res.get("statefarm_available", 0),
            "road": road_obj_res.get("road_available", 0),
            "geometry": road_geom_res.get("geometry_available", 0)
        }
        
        # Road score
        road_score = self.compute_road_relevance(road_obj_res)
        
        # Combine all features matching exactly what the model expects
        current_features = {
            "drowsy_drowsy_prob": drowsiness_res.get("drowsy_prob", 0.0),
            "drowsy_not_drowsy_prob": drowsiness_res.get("not_drowsy_prob", 0.0),
            "drowsy_alert_prob": drowsiness_res.get("alert_prob", 0.0),
            "drowsy_microsleep_prob": drowsiness_res.get("microsleep_prob", 0.0),
            "drowsy_yawning_prob": drowsiness_res.get("yawning_prob", 0.0),
            "drowsy_prediction_confidence": drowsiness_res.get("prediction_confidence", 0.0),
            
            "sf_drinking_prob": statefarm_res.get("sf_drinking_prob", 0.0),
            "sf_normal_prob": statefarm_res.get("sf_normal_prob", 0.0),
            "sf_phone_prob": statefarm_res.get("sf_phone_prob", 0.0),
            "sf_radio_prob": statefarm_res.get("sf_radio_prob", 0.0),
            "sf_reaching_prob": statefarm_res.get("sf_reaching_prob", 0.0),
            "sf_passenger_prob": statefarm_res.get("sf_passenger_prob", 0.0),
            "sf_texting_prob": statefarm_res.get("sf_texting_prob", 0.0),
            "sf_behavior_max_prob": statefarm_res.get("sf_behavior_max_prob", 0.0),
            "sf_distraction_prob": statefarm_res.get("sf_distraction_prob", 0.0),
            
            "road_object_count": road_obj_res.get("object_count", 0),
            "road_car_count": road_obj_res.get("car_count", 0),
            "road_truck_count": road_obj_res.get("truck_count", 0),
            "road_bus_count": road_obj_res.get("bus_count", 0),
            "road_two_wheeler_count": road_obj_res.get("two_wheeler_count", 0),
            "road_person_count": road_obj_res.get("person_count", 0),
            "road_max_object_confidence": road_obj_res.get("max_object_confidence", 0.0),
            "road_nearest_object_confidence": road_obj_res.get("nearest_object_confidence", 0.0),
            "road_nearest_object_area_ratio": road_obj_res.get("nearest_object_area_ratio", 0.0),
            "road_drivable_fraction": road_geom_res.get("drivable_fraction", 0.0),
            "road_lane_fraction": road_geom_res.get("lane_fraction", 0.0),
            "road_road_relevance_score": road_score,
            
            "drowsiness_available": avail["drowsiness"],
            "statefarm_available": avail["statefarm"],
            "road_available": avail["road"]
        }
        
        self.history.append(current_features)
        
        # Replace NaN/Inf
        for k, v in current_features.items():
            if np.isnan(v) or np.isinf(v):
                current_features[k] = 0.0
                
        # Build vector exactly matching schema order
        vector = [current_features.get(name, 0.0) for name in self.feature_names]
        dmatrix = xgb.DMatrix(np.array([vector]), feature_names=self.feature_names)
        
        events_output = {}
        decision_mode = "FULL" if all(avail.values()) else "DEGRADED"
        
        for event, model in self.models.items():
            prob = float(model.predict(dmatrix)[0])
            
            # Map event to config names
            cfg_name = f"event_{event}"
            threshold = self.threshold_mgr.get_threshold(cfg_name)
            req_mod = self.threshold_mgr.get_required_modality(cfg_name)
            
            # Check availability
            is_active = None
            state_str = "NOT_OBSERVED"
            
            if req_mod and avail.get(req_mod.replace("_available", ""), 0) == 0:
                state_str = "NOT_OBSERVED"
                is_active = None
            else:
                is_active = bool(prob >= threshold)
                state_str = "ACTIVE" if is_active else "INACTIVE"
                
            events_output[event] = {
                "probability": prob,
                "threshold": threshold,
                "active": is_active,
                "state": state_str
            }
            
        observed_count = sum(1 for e in events_output.values() if e["active"])
        if decision_mode == "FULL":
            is_normal = (observed_count == 0)
        else:
            is_normal = False if observed_count > 0 else None
        
        return {
            "timestamp": timestamp,
            "decision_mode": decision_mode,
            "availability": {
                "drowsiness": bool(avail["drowsiness"]),
                "statefarm": bool(avail["statefarm"]),
                "road": bool(avail["road"])
            },
            "events": events_output,
            "normal": is_normal,
            "observed_event_count": observed_count
        }
