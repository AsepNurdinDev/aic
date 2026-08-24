import json

class ThresholdManager:
    def __init__(self, config_path: str):
        self.config = {}
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Failed to load thresholds from {config_path}: {e}")
            
    def get_threshold(self, event_name: str) -> float:
        # e.g. "event_drowsiness"
        if event_name in self.config:
            return float(self.config[event_name].get("threshold", 0.5))
        return 0.5
        
    def get_required_modality(self, event_name: str) -> str:
        if event_name in self.config:
            return self.config[event_name].get("required_modality", "")
        return ""
