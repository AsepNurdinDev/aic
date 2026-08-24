# pyrefly: ignore [missing-import]
import pytest
from src.decision.safety_policy import SafetyPolicy

def test_safety_policy_full_mode():
    policy = SafetyPolicy()
    
    # 0 events -> SAFE
    decision = {
        "decision_mode": "FULL",
        "observed_event_count": 0,
        "events": {}
    }
    result = policy.evaluate(decision)
    assert result["severity"] == "SAFE"
    assert result["action"] == "NONE"
    
    # 3 events -> CRITICAL
    decision = {
        "decision_mode": "FULL",
        "observed_event_count": 3,
        "events": {
            "drowsiness": {"active": True},
            "phone_use": {"active": True},
            "road_risk": {"active": True}
        }
    }
    result = policy.evaluate(decision)
    assert result["severity"] == "CRITICAL"
    assert result["action"] == "URGENT_WARNING"

def test_safety_policy_degraded_mode():
    policy = SafetyPolicy()
    
    # 0 events in degraded mode -> DEGRADED / DEGRADED_WARNING
    decision = {
        "decision_mode": "DEGRADED",
        "observed_event_count": 0,
        "events": {}
    }
    result = policy.evaluate(decision)
    assert result["severity"] == "DEGRADED"
    assert result["action"] == "DEGRADED_WARNING"
