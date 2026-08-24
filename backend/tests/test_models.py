# pyrefly: ignore [missing-import]
import pytest
from src.perception.drowsiness import DrowsinessProcessor
from src.perception.statefarm import StateFarmProcessor
from src.perception.road_object import RoadObjectProcessor
from src.perception.road_geometry import RoadGeometryProcessor

def test_models_importable():
    assert DrowsinessProcessor is not None
    assert StateFarmProcessor is not None
    assert RoadObjectProcessor is not None
    assert RoadGeometryProcessor is not None
