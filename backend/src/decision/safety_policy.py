class SafetyPolicy:
    def __init__(self):
        pass
        
    def evaluate(self, decision_output: dict) -> dict:
        mode = decision_output["decision_mode"]
        observed = decision_output["observed_event_count"]
        events = decision_output["events"]
        
        # Determine Severity
        if mode == "FULL":
            if observed == 0:
                severity = "SAFE"
            elif observed == 1:
                severity = "CAUTION"
            elif observed == 2:
                severity = "HIGH"
            else:
                severity = "CRITICAL"
        else:
            if observed == 0:
                severity = "DEGRADED"
            elif observed == 1:
                severity = "CAUTION"
            elif observed == 2:
                severity = "HIGH"
            else:
                severity = "CRITICAL"
                
        # Determine base action by priority
        action = "NONE"
        
        is_drowsy = events.get("drowsiness", {}).get("active", False)
        
        # Check distraction (anything from statefarm except normal)
        distractions = ["phone_use", "texting", "drinking", "radio_operation", "reaching_behind", "talking_passenger"]
        is_distracted = any(events.get(e, {}).get("active", False) for e in distractions)
        
        is_road_risk = events.get("road_risk", {}).get("active", False)
        
        if is_drowsy:
            action = "DROWSINESS_WARNING"
        elif is_distracted:
            action = "DISTRACTION_WARNING"
        elif is_road_risk:
            action = "ROAD_WARNING"
            
        # Overrides based on severity and mode
        if severity == "CRITICAL":
            action = "URGENT_WARNING"
        elif mode == "DEGRADED" and observed == 0:
            action = "DEGRADED_WARNING"
            
        decision_output["severity"] = severity
        decision_output["action"] = action
        
        return decision_output
