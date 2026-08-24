ENABLE_LOGGING = True
EVENT_CONFIRM_FRAMES = 3
EVENT_CLEAR_FRAMES = 5
WARNING_COOLDOWN_SEC = 5.0

DROWSINESS_MODEL_PATH = "models/perception/fl3d_v5_blinkaware_cnn_bigru_best.pth"
LANDMARKER_PATH = "models/perception/face_landmarker.task"
STATEFARM_MODEL_PATH = "models/perception/convnext_tiny_statefarm_7class_best.pth"
ROAD_OBJECT_MODEL_PATH = "models/perception/road_object_yolo26n_best.pt"
ROAD_GEOMETRY_MODEL_PATH = "models/perception/road_geometry_v2_best.pt"
DECISION_MODELS_DIR = "models/decision"

# Schema path (ensure this exists or adjust accordingly)
SCHEMA_PATH = "aic/backend/configs/decision_feature_schema.json"
