import time

class WarningManager:
    def __init__(self, cooldown_sec=5.0):
        self.cooldown_sec = cooldown_sec
        self.last_warning_time = 0.0
        self.last_message = ""
        self.last_action = "NONE"
        
    def generate_message(self, decision: dict) -> str:
        severity = decision.get("severity", "SAFE")
        action = decision.get("action", "NONE")
        events = decision.get("events", {})
        
        is_drowsy = events.get("drowsiness", {}).get("active", False)
        
        distractions = ["phone_use", "texting", "drinking", "radio_operation", "reaching_behind", "talking_passenger"]
        is_distracted = any(events.get(e, {}).get("active", False) for e in distractions)
        is_phone = events.get("phone_use", {}).get("active", False) or events.get("texting", {}).get("active", False)
        
        is_road_risk = events.get("road_risk", {}).get("active", False)
        
        message = ""
        
        if severity == "CRITICAL" and action == "URGENT_WARNING":
            if is_drowsy and is_phone:
                message = "Anda mengantuk dan sedang menggunakan ponsel. Harap segera berhenti dan fokus pada jalan."
            elif is_drowsy and is_distracted:
                message = "Anda mengantuk dan sedang terdistraksi. Harap segera berhenti dan fokus pada jalan."
            else:
                message = "Peringatan Kritis! Segera tingkatkan kewaspadaan Anda."
        elif action == "DROWSINESS_WARNING":
            message = "Anda terlihat mengantuk. Harap berhenti dan beristirahat."
        elif action == "DISTRACTION_WARNING":
            if is_phone:
                message = "Harap hentikan penggunaan ponsel dan fokus pada jalan."
            else:
                message = "Anda terdistraksi. Harap fokus pada jalan."
        elif action == "ROAD_WARNING":
            message = "Risiko jalan terdeteksi. Harap tingkatkan kewaspadaan."
        elif action == "DEGRADED_WARNING":
            message = "Sistem monitoring sebagian tidak tersedia. Harap tetap berhati-hati."
            
        return message
        
    def process(self, decision: dict) -> str | None:
        """Process decision and return warning message if one should be emitted.
        
        Returns the warning message string, or None if no warning should be played
        (due to cooldown or no warning needed).
        """
        message = self.generate_message(decision)
        current_time = time.time()
        current_action = decision.get("action", "NONE")
        
        if message:
            # Play if action changed OR cooldown expired
            action_changed = current_action != self.last_action
            cooldown_expired = (current_time - self.last_warning_time) > self.cooldown_sec
            
            if action_changed or cooldown_expired:
                print(f"[VOICE/WARNING] {message}")
                self.last_warning_time = current_time
                self.last_message = message
                self.last_action = current_action
                return message
        else:
            self.last_message = ""
            self.last_action = current_action
            
        return None
