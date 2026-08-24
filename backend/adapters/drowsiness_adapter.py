from typing import Dict, Any
from fusion.feature_schema import DrowsinessOutput

class DrowsinessAdapter:
    """
    Adapter to convert V5 Drowsiness output into standardized DrowsinessOutput.
    Note: The current V5 backend does not output specific class probabilities (alert, microsleep, yawning)
    or state durations in its API response, so those fields are omitted/defaulted.
    """
    
    @staticmethod
    def adapt(raw_output: Dict[str, Any]) -> DrowsinessOutput:
        if not raw_output or raw_output.get("status") in ["UNKNOWN", "NO FACE", "INVALID FEATURES", None]:
            return DrowsinessOutput(drowsiness_available=0)
            
        status = raw_output.get("status", "UNKNOWN")
        scenario = raw_output.get("scenario", "UNKNOWN")
        raw_prob = float(raw_output.get("raw_prob", 0.0))
        alarm_active = int(raw_output.get("alarm_active", False))
        
        # In the current implementation, we only get the drowsy probability.
        # We don't get exact durations or multi-class probabilities.
        return DrowsinessOutput(
            drowsiness_available=1,
            drowsy_prob=raw_prob,
            not_drowsy_prob=1.0 - raw_prob if raw_prob is not None else 0.0,
            drowsiness_state=scenario,
            is_drowsy=1 if status == "DROWSY" else 0,
            alarm_active=alarm_active
            # Note: alert_prob, microsleep_prob, yawning_prob, drowsy_duration, etc. 
            # are removed from schema adaptation as they are not provided by the current model output.
        )
