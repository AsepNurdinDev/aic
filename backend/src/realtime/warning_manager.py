import time

class WarningManager:
    def __init__(self, cooldown_sec=5.0):
        self.cooldown_sec = cooldown_sec
        self.last_warning_time = 0.0
        self.last_message = ""
        
    def generate_message(self, decision: dict) -> str:
        severity = decision.get("severity", "SAFE")
        action = decision.get("action", "NONE")
        events = decision.get("events", {})
        
        is_drowsy = events.get("drowsiness", {}).get("active", False)
        
        distractions = ["phone_use", "texting", "drinking", "radio_operation", "reaching_behind", "talking_passenger"]
        is_distracted = any(events.get(e, {}).get("active", False) for e in distractions)
        
        is_road_risk = events.get("road_risk", {}).get("active", False)
        
        message = ""
        
        if severity == "CRITICAL" and action == "URGENT_WARNING":
            if is_drowsy and is_distracted:
                message = "Anda mengantuk dan sedang terdistraksi. Harap segera berhenti dan fokus pada jalan."
            else:
                message = "Peringatan Kritis! Segera tingkatkan kewaspadaan Anda."
        elif action == "DROWSINESS_WARNING":
            message = "Anda terlihat mengantuk. Harap berhenti dan beristirahat."
        elif action == "DISTRACTION_WARNING":
            if events.get("phone_use", {}).get("active", False) or events.get("texting", {}).get("active", False):
                message = "Harap hentikan penggunaan ponsel dan fokus pada jalan."
            else:
                message = "Anda terdistraksi. Harap fokus pada jalan."
        elif action == "ROAD_WARNING":
            message = "Risiko jalan terdeteksi. Harap tingkatkan kewaspadaan."
        elif action == "DEGRADED_WARNING":
            message = "Sistem monitoring sebagian tidak tersedia. Harap tetap berhati-hati."
            
        return message
        
    def process(self, decision: dict):
        message = self.generate_message(decision)
        current_time = time.time()
        
        if message and message != "NONE":
            # Cooldown logic
            if (current_time - self.last_warning_time) > self.cooldown_sec or message != self.last_message:
                print(f"[VOICE/WARNING] {message}")
                self.last_warning_time = current_time
                self.last_message = message
        else:
            self.last_message = ""
