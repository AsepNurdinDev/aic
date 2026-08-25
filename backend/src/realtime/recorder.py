import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
RECORDINGS_DIR = Path(__file__).resolve().parent.parent.parent / "recordings"

class SessionRecorder:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RECORDINGS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = False
        self.session_id: Optional[str] = None
        self.start_time: Optional[str] = None
        self.records: List[Dict[str, Any]] = []
        self.frame_counter = 0
        self.latest_saved_file: Optional[Path] = None

    def start_session(self, session_name: Optional[str] = None) -> str:
        """Start a new recording session."""
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
        prefix = session_name if session_name else "session"
        self.session_id = f"{prefix}_{timestamp_str}"
        self.start_time = now_dt.isoformat()
        self.records = []
        self.frame_counter = 0
        self.is_recording = True
        return self.session_id

    def record_frame(self, 
                     inputs_meta: Dict[str, Any], 
                     decision: Dict[str, Any]):
        """Record input metadata and full inference outputs for one frame."""
        if not self.is_recording:
            # If not explicitly started, we still maintain a sliding session buffer
            if self.session_id is None:
                self.start_session(session_name="auto_session")

        self.frame_counter += 1
        
        record = {
            "frame_id": self.frame_counter,
            "timestamp": decision.get("timestamp", time.time()),
            "datetime": datetime.now().isoformat(),
            "inputs": {
                "driver_frame_received": bool(inputs_meta.get("driver_frame_received")),
                "road_frame_received": bool(inputs_meta.get("road_frame_received")),
                "cabin_frame_received": bool(inputs_meta.get("cabin_frame_received")),
                "perception_outputs": decision.get("perception_inputs", {})
            },
            "outputs": {
                "decision_mode": decision.get("decision_mode", "DEGRADED"),
                "availability": decision.get("availability", {}),
                "events": decision.get("events", {}),
                "observed_event_count": decision.get("observed_event_count", 0),
                "normal": decision.get("normal", True),
                "severity": decision.get("severity", "SAFE"),
                "action": decision.get("action", "NONE"),
                "warning_message": decision.get("warning_message")
            }
        }
        
        self.records.append(record)

    def stop_session(self) -> Dict[str, Any]:
        """Stop current session and persist to a JSON file."""
        if not self.records and not self.session_id:
            return {"status": "NO_ACTIVE_SESSION", "records_count": 0}

        end_time = datetime.now().isoformat()
        
        # Aggregate statistics for quick insights
        severity_counts = {}
        action_counts = {}
        event_trigger_counts = {}
        
        for r in self.records:
            sev = r["outputs"].get("severity", "UNKNOWN")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            act = r["outputs"].get("action", "NONE")
            if act != "NONE":
                action_counts[act] = action_counts.get(act, 0) + 1
                
            for ev_name, ev_data in r["outputs"].get("events", {}).items():
                if ev_data.get("active") is True:
                    event_trigger_counts[ev_name] = event_trigger_counts.get(ev_name, 0) + 1

        payload = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": end_time,
            "total_frames": len(self.records),
            "summary": {
                "severity_counts": severity_counts,
                "triggered_actions": action_counts,
                "event_triggers": event_trigger_counts
            },
            "records": self.records
        }

        filename = f"{self.session_id}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, cls=NpEncoder)

        self.latest_saved_file = filepath
        self.is_recording = False
        
        return {
            "status": "SAVED",
            "filename": filename,
            "filepath": str(filepath),
            "total_frames": len(self.records),
            "summary": payload["summary"]
        }

    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get latest session in-memory data or from the latest saved file."""
        if self.records:
            return {
                "session_id": self.session_id,
                "start_time": self.start_time,
                "total_frames": len(self.records),
                "records": self.records
            }
        if self.latest_saved_file and self.latest_saved_file.exists():
            with open(self.latest_saved_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """List all saved session JSON files in the recordings directory."""
        results = []
        for file in sorted(self.output_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
            try:
                stat = file.stat()
                results.append({
                    "filename": file.name,
                    "filepath": str(file),
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception:
                pass
        return results
