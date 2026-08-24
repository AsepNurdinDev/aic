import time
import cv2
import numpy as np

from src.perception.drowsiness import DrowsinessProcessor
from src.perception.statefarm import StateFarmProcessor
from src.perception.road_object import RoadObjectProcessor
from src.perception.road_geometry import RoadGeometryProcessor

from src.decision.decision_engine import DecisionEngine
from src.decision.safety_policy import SafetyPolicy

from src.realtime.smoothing import DecisionSmoother
from src.realtime.warning_manager import WarningManager

from src.utils.config import *

class RealtimePipeline:
    def __init__(self):
        print("==================================================")
        print("MODEL STATUS")
        print("==================================================")
        
        # Load Drowsiness Model
        try:
            self.drowsiness_processor = DrowsinessProcessor(DROWSINESS_MODEL_PATH, LANDMARKER_PATH)
            print("Drowsiness model: READY")
        except Exception as e:
            print(f"Drowsiness model: FAILED ({e})")
            self.drowsiness_processor = None
            
        # Load State Farm Model
        try:
            self.statefarm_processor = StateFarmProcessor(STATEFARM_MODEL_PATH)
            print("State Farm model: READY")
        except Exception as e:
            print(f"State Farm model: FAILED ({e})")
            self.statefarm_processor = None
            
        # Load Road Object Model
        try:
            self.road_object_processor = RoadObjectProcessor(ROAD_OBJECT_MODEL_PATH)
            print("Road object model: READY")
        except Exception as e:
            print(f"Road object model: FAILED ({e})")
            self.road_object_processor = None
            
        # Load Road Geometry Model
        try:
            self.road_geometry_processor = RoadGeometryProcessor(ROAD_GEOMETRY_MODEL_PATH)
            print("Road geometry model: READY")
        except Exception as e:
            print(f"Road geometry model: FAILED ({e})")
            self.road_geometry_processor = None

        # Load Decision AI
        try:
            self.decision_engine = DecisionEngine(DECISION_MODELS_DIR, SCHEMA_PATH)
            print("Decision AI: READY")
        except Exception as e:
            print(f"Decision AI: FAILED ({e})")
            self.decision_engine = None
            
        self.safety_policy = SafetyPolicy()
        
        self.smoother = DecisionSmoother(confirm_frames=EVENT_CONFIRM_FRAMES, clear_frames=EVENT_CLEAR_FRAMES)
        self.warning_manager = WarningManager(cooldown_sec=WARNING_COOLDOWN_SEC)
        
        self.last_time = time.time()
        print("==================================================\n")
        
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """Process a single frame through all perception models.
        Used when all cameras share the same frame (legacy/fallback)."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._run_inference(
            driver_rgb=frame_rgb,
            road_rgb=frame_rgb,
            cabin_rgb=None
        )
    
    def process_channels(self, 
                         driver_frame: np.ndarray | None = None, 
                         road_frame: np.ndarray | None = None, 
                         cabin_frame: np.ndarray | None = None) -> dict:
        """Process separate frames from different camera channels.
        
        Args:
            driver_frame: BGR frame from driver camera (for drowsiness + statefarm)
            road_frame: BGR frame from road camera (for road_object + road_geometry)
            cabin_frame: BGR frame from cabin camera (reserved for future)
            
        Returns:
            Decision dict with warning_message included.
        """
        driver_rgb = cv2.cvtColor(driver_frame, cv2.COLOR_BGR2RGB) if driver_frame is not None else None
        road_rgb = cv2.cvtColor(road_frame, cv2.COLOR_BGR2RGB) if road_frame is not None else None
        cabin_rgb = cv2.cvtColor(cabin_frame, cv2.COLOR_BGR2RGB) if cabin_frame is not None else None
        
        return self._run_inference(driver_rgb, road_rgb, cabin_rgb)
    
    def _run_inference(self, 
                       driver_rgb: np.ndarray | None, 
                       road_rgb: np.ndarray | None, 
                       cabin_rgb: np.ndarray | None) -> dict:
        """Core inference pipeline — routes frames to the correct perception models."""
        current_time = time.time()
        
        # Perception — Drowsiness (driver camera)
        if driver_rgb is not None and self.drowsiness_processor:
            try:
                drowsiness_res = self.drowsiness_processor.process(driver_rgb)
            except Exception:
                drowsiness_res = {"drowsiness_available": 0}
        else:
            drowsiness_res = {"drowsiness_available": 0}
            
        # Perception — State Farm (driver camera)
        if driver_rgb is not None and self.statefarm_processor:
            try:
                statefarm_res = self.statefarm_processor.process(driver_rgb)
            except Exception:
                statefarm_res = {"statefarm_available": 0}
        else:
            statefarm_res = {"statefarm_available": 0}
            
        # Perception — Road Object (road camera)
        if road_rgb is not None and self.road_object_processor:
            try:
                road_obj_res = self.road_object_processor.process(road_rgb)
            except Exception:
                road_obj_res = {"road_available": 0}
        else:
            road_obj_res = {"road_available": 0}
            
        # Perception — Road Geometry (road camera)
        if road_rgb is not None and self.road_geometry_processor:
            try:
                road_geom_res = self.road_geometry_processor.process(road_rgb)
            except Exception:
                road_geom_res = {"geometry_available": 0}
        else:
            road_geom_res = {"geometry_available": 0}
        
        # Decision
        if self.decision_engine:
            decision = self.decision_engine.process(
                current_time, 
                drowsiness_res, 
                statefarm_res, 
                road_obj_res, 
                road_geom_res
            )
        else:
            decision = {
                "timestamp": current_time,
                "decision_mode": "DEGRADED",
                "events": {},
                "observed_event_count": 0,
                "normal": True
            }
            
        # Smoothing
        decision = self.smoother.process(decision)
        
        # Safety Policy
        decision = self.safety_policy.evaluate(decision)
        
        # Warning Message
        warning_msg = self.warning_manager.process(decision)
        decision["warning_message"] = warning_msg
        
        return decision
