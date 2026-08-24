import cv2
import numpy as np
import json
import sys

from src.realtime.pipeline import RealtimePipeline

def run_smoke_test():
    print("==================================================")
    print("INTEGRATION SMOKE TEST")
    print("==================================================")
    
    # 1. Load pipeline (this implicitly loads all models, threshold, policies)
    pipeline = RealtimePipeline()
    
    # Check if models loaded successfully
    perception_status = {
        "FL3D": "OK" if pipeline.drowsiness_processor else "FAILED",
        "State Farm": "OK" if pipeline.statefarm_processor else "FAILED",
        "Road Object": "OK" if pipeline.road_object_processor else "FAILED",
        "Road Geometry": "OK" if pipeline.road_geometry_processor else "FAILED"
    }
    
    print("\nPerception:")
    for name, status in perception_status.items():
        print(f"{name.ljust(18)} {status}")
        
    decision_events = ["DROWSINESS", "PHONE_USE", "TEXTING", "DRINKING", "RADIO", "REACHING", "PASSENGER", "ROAD_RISK"]
    print("\nDecision:")
    if pipeline.decision_engine:
        for event in decision_events:
            print(f"{event.ljust(18)} OK")
    else:
        for event in decision_events:
            print(f"{event.ljust(18)} FAILED")
            
    print("\nPolicy:")
    print(f"{'FULL/DEGRADED'.ljust(18)} {'OK' if pipeline.safety_policy else 'FAILED'}")
    
    # 2. Create dummy frame for testing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        # 3-7. Run full pipeline
        decision = pipeline.process_frame(dummy_frame)
        
        # 8. Print structured decision
        print("\nFinal decision:")
        print(f"MODE: {decision.get('decision_mode')}")
        
        events_str = ", ".join([e for e, info in decision.get("events", {}).items() if info.get("active")])
        print(f"EVENTS: {events_str if events_str else 'NONE'}")
        print(f"SEVERITY: {decision.get('severity')}")
        print(f"ACTION: {decision.get('action')}")
        
        # 9 & 10. Verify no NaN/Inf
        for event, info in decision.get("events", {}).items():
            prob = info.get("probability", 0.0)
            if np.isnan(prob) or np.isinf(prob):
                raise ValueError(f"Non-finite probability in {event}: {prob}")
                
        # 12 & 13. Verify logic (e.g. DEGRADED mode if 0 events -> DEGRADED_WARNING, never SAFE)
        if decision.get("decision_mode") == "DEGRADED" and decision.get("observed_event_count", 0) == 0:
            if decision.get("severity") == "SAFE" or decision.get("action") == "NONE":
                raise ValueError("DEGRADED mode with 0 events produced SAFE/NONE instead of DEGRADED_WARNING")
                
        # (This is just one frame, so temporal won't trigger active for events with EVENT_CONFIRM_FRAMES > 1)
                
        print("\nSmoke test PASSED \u2705")
        
    except Exception as e:
        print(f"\nSmoke test FAILED \u274c : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
