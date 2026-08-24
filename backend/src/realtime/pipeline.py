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

from src.utils.logger import CsvLogger
from src.utils.config import *

class RealtimePipeline:
    def __init__(self):
        print("==================================================")
        print("MODEL STATUS")
        print("==================================================")
        
        # Load Drowsiness Model
        try:
            self.drowsiness_processor = DrowsinessProcessor(DROWSINESS_MODEL_PATH, LANDMARKER_PATH)
            print("FL3D:\nLOADED\n")
        except Exception as e:
            print(f"FL3D:\nFAILED ({e})\n")
            self.drowsiness_processor = None
            
        # Load State Farm Model
        try:
            self.statefarm_processor = StateFarmProcessor(STATEFARM_MODEL_PATH)
            print("State Farm:\nLOADED\n")
        except Exception as e:
            print(f"State Farm:\nFAILED ({e})\n")
            self.statefarm_processor = None
            
        # Load Road Object Model
        try:
            self.road_object_processor = RoadObjectProcessor(ROAD_OBJECT_MODEL_PATH)
            print("Road Object:\nLOADED\n")
        except Exception as e:
            print(f"Road Object:\nFAILED ({e})\n")
            self.road_object_processor = None
            
        # Load Road Geometry Model
        try:
            self.road_geometry_processor = RoadGeometryProcessor(ROAD_GEOMETRY_MODEL_PATH)
            print("Road Geometry:\nLOADED\n")
        except Exception as e:
            print(f"Road Geometry:\nFAILED ({e})\n")
            self.road_geometry_processor = None

        # Load Decision AI
        try:
            self.decision_engine = DecisionEngine(DECISION_MODELS_DIR, SCHEMA_PATH)
            print("Decision:")
            for event in ["DROWSINESS", "PHONE", "TEXTING", "DRINKING", "RADIO", "REACHING", "PASSENGER", "ROAD"]:
                print(f"{event} LOADED")
            print("\nThresholds:\nLOADED\n")
        except Exception as e:
            print(f"Decision System:\nFAILED ({e})\n")
            self.decision_engine = None
            
        self.safety_policy = SafetyPolicy()
        print("Policy:\nLOADED\n")
        print("==================================================\n")
        
        self.smoother = DecisionSmoother(confirm_frames=EVENT_CONFIRM_FRAMES, clear_frames=EVENT_CLEAR_FRAMES)
        self.warning_manager = WarningManager(cooldown_sec=WARNING_COOLDOWN_SEC)
        self.logger = CsvLogger(enabled=ENABLE_LOGGING)
        
        self.last_time = time.time()
        
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0.0
        self.last_time = current_time
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Perception
        drowsiness_res = self.drowsiness_processor.process(frame_rgb) if self.drowsiness_processor else {"drowsiness_available": 0}
        statefarm_res = self.statefarm_processor.process(frame_rgb) if self.statefarm_processor else {"statefarm_available": 0}
        road_obj_res = self.road_object_processor.process(frame_rgb) if self.road_object_processor else {"road_available": 0}
        road_geom_res = self.road_geometry_processor.process(frame_rgb) if self.road_geometry_processor else {"geometry_available": 0}
        
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
            # Fallback if engine fails to load
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
        
        # Warning & Logging
        self.warning_manager.process(decision)
        self.logger.log(decision, fps)
        
        return decision
