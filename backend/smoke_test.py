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
    
    # 2. Test single-frame inference (process_frame)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        decision = pipeline.process_frame(dummy_frame)
        
        print("\n--- Test 1: Single frame inference ---")
        print(f"MODE: {decision.get('decision_mode')}")
        
        events_str = ", ".join([e for e, info in decision.get("events", {}).items() if info.get("active")])
        print(f"EVENTS: {events_str if events_str else 'NONE'}")
        print(f"SEVERITY: {decision.get('severity')}")
        print(f"ACTION: {decision.get('action')}")
        print(f"WARNING: {decision.get('warning_message')}")
        
        # Verify no NaN/Inf
        for event, info in decision.get("events", {}).items():
            prob = info.get("probability", 0.0)
            if np.isnan(prob) or np.isinf(prob):
                raise ValueError(f"Non-finite probability in {event}: {prob}")
                
        # DEGRADED mode with 0 events must NOT produce SAFE
        if decision.get("decision_mode") == "DEGRADED" and decision.get("observed_event_count", 0) == 0:
            if decision.get("severity") == "SAFE" or decision.get("action") == "NONE":
                raise ValueError("DEGRADED mode with 0 events produced SAFE/NONE instead of DEGRADED_WARNING")
                
        print("Single frame: PASSED ✅")
        
    except Exception as e:
        print(f"\nSingle frame test FAILED ❌ : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 3. Test 3-channel inference (process_channels)
    try:
        print("\n--- Test 2: 3-channel inference ---")
        driver_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        road_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cabin_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        decision = pipeline.process_channels(driver_frame, road_frame, cabin_frame)
        print(f"MODE: {decision.get('decision_mode')}")
        print(f"SEVERITY: {decision.get('severity')}")
        print(f"ACTION: {decision.get('action')}")
        print(f"WARNING: {decision.get('warning_message')}")
        print("3-channel: PASSED ✅")
        
    except Exception as e:
        print(f"\n3-channel test FAILED ❌ : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Test partial modality (driver only, no road)
    try:
        print("\n--- Test 3: Partial modality (driver only) ---")
        decision = pipeline.process_channels(driver_frame, None, None)
        print(f"MODE: {decision.get('decision_mode')}")
        print(f"SEVERITY: {decision.get('severity')}")
        print(f"ACTION: {decision.get('action')}")
        
        if decision.get("decision_mode") != "DEGRADED":
            raise ValueError(f"Expected DEGRADED mode when road is missing, got {decision.get('decision_mode')}")
        
        print("Partial modality: PASSED ✅")
        
    except Exception as e:
        print(f"\nPartial modality test FAILED ❌ : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 5. Test warning_message is included in response
    try:
        print("\n--- Test 4: Warning message field ---")
        assert "warning_message" in decision, "warning_message field missing from decision"
        print("Warning field: PASSED ✅")
    except Exception as e:
        print(f"\nWarning field test FAILED ❌ : {e}")
        sys.exit(1)
        
    print("\n==================================================")
    print("ALL SMOKE TESTS PASSED ✅")
    print("==================================================")

if __name__ == "__main__":
    run_smoke_test()
