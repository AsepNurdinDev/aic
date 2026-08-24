import csv
import os
import time

class CsvLogger:
    def __init__(self, filepath="runtime_decisions.csv", enabled=True):
        self.filepath = filepath
        self.enabled = enabled
        self.initialized = False
        
    def _init_file(self, decision: dict):
        if not self.enabled or self.initialized:
            return
            
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = [
                "timestamp", "fps", "decision_mode", "severity", "action",
                "drowsiness_avail", "statefarm_avail", "road_avail"
            ]
            for event_name in decision.get("events", {}).keys():
                headers.append(f"{event_name}_prob")
                headers.append(f"{event_name}_state")
            writer.writerow(headers)
        self.initialized = True
        
    def log(self, decision: dict, fps: float):
        if not self.enabled:
            return
            
        if not self.initialized:
            self._init_file(decision)
            
        row = [
            decision.get("timestamp", time.time()),
            f"{fps:.1f}",
            decision.get("decision_mode", ""),
            decision.get("severity", ""),
            decision.get("action", ""),
            decision.get("availability", {}).get("drowsiness", False),
            decision.get("availability", {}).get("statefarm", False),
            decision.get("availability", {}).get("road", False)
        ]
        
        for event_info in decision.get("events", {}).values():
            row.append(f"{event_info.get('probability', 0.0):.4f}")
            row.append(event_info.get("state", "UNKNOWN"))
            
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
