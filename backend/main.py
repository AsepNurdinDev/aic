import cv2
import argparse
from src.realtime.pipeline import RealtimePipeline

def main():
    parser = argparse.ArgumentParser(description="Realtime Driver Monitoring System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()
    
    pipeline = RealtimePipeline()
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
        
    print(f"Starting realtime monitoring on camera {args.camera}. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        decision = pipeline.process_frame(frame)
        
        # Optional: draw some info on the frame for debugging
        mode = decision.get("decision_mode", "")
        severity = decision.get("severity", "")
        action = decision.get("action", "")
        fps_val = pipeline.logger.initialized # just as a placeholder since pipeline encapsulates logger
        
        cv2.putText(frame, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"SEVERITY: {severity}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if severity == "CRITICAL" else (0, 255, 255), 2)
        cv2.putText(frame, f"ACTION: {action}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Realtime AI Driver Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
