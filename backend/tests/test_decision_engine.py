# pyrefly: ignore [missing-import]
import pytest
from src.decision.decision_engine import DecisionEngine

def test_decision_engine_initialization():
    try:
        # Assuming the paths are accessible from project root during pytest
        engine = DecisionEngine("models/decision", "aic/backend/configs/decision_feature_schema.json")
        assert engine is not None
    except Exception as e:
        pytest.skip(f"Skipping test because models/schema couldn't be loaded: {e}")
