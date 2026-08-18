"""
backend/src/driver_analyzer.py

Pipeline lengkap analisis frame pengemudi real-time:
1. MediaPipe Tasks Face Landmarker
2. Fitur EAR V3 + MAR
3. Sliding Sequence Buffer (60 frame) + Z-score Normalization
4. LandmarkGRU PyTorch Model Inference
5. Temporal Prediction Smoother
6. Head Pose Estimator (Nodding Pitch & Look-Aside Yaw)
7. Skenario Fisiologis Realistis (Microsleep, Nodding, Looking Aside, Yawning, Drowsy GRU, Normal)
8. Fatigue Timer Tracker & Alert Thresholds
9. SafeRoute AI Risk Engine (LOW, POTENTIAL, WARNING, CRITICAL)
"""

import time
import math
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

from src.config import (
    MODEL_PATH,
    CONFIG_PATH,
    MEDIAPIPE_MODEL_PATH,
    SEQUENCE_LENGTH,
    THRESHOLD,
    SMOOTHING_WINDOW,
    DROWSY_ALERT_SECONDS,
    NODDING_ALERT_SECONDS,
    LOOK_ASIDE_ALERT_SECONDS,
    MICROSLEEP_ALERT_SECONDS,
    YAWN_ALERT_SECONDS,
    EYE_CLOSED_EAR_THRESHOLD,
    YAWN_MAR_THRESHOLD,
    NODDING_PITCH_RATIO_THRESHOLD,
    LOOK_ASIDE_YAW_THRESHOLD,
    get_feature_stats,
)
from src.landmark_detector import FaceLandmarkDetector
from src.feature_extractor import (
    extract_features,
    evaluate_head_pose,
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH,
    LM_FOREHEAD,
    LM_NOSE,
    LM_CHIN,
    LM_LEFT_CHEEK,
    LM_RIGHT_CHEEK,
)
from src.preprocessing import SequenceBuffer, load_training_statistics
from src.inference import DrowsinessInference
from src.smoothing import PredictionSmoother
from src.risk_engine import RiskEngine


class DriverAnalyzer:
    """
    Stateful Analyzer untuk memproses stream video pengemudi secara real-time.
    """

    def __init__(
        self,
        checkpoint_path: str = MODEL_PATH,
        config_path: str = CONFIG_PATH,
        mediapipe_model_path: str = MEDIAPIPE_MODEL_PATH,
        device: str = "cpu"
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.mediapipe_model_path = mediapipe_model_path
        self.device = device

        # 1. Detector
        self.detector = FaceLandmarkDetector(model_path=mediapipe_model_path)

        # 2. Model & Stats
        self.engine = DrowsinessInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            default_threshold=THRESHOLD
        )

        mean, std, seq_len = load_training_statistics(
            checkpoint_path=checkpoint_path,
            config_path=config_path
        )
        self.sequence_length = seq_len
        self.buffer = SequenceBuffer(
            sequence_length=seq_len,
            feature_mean=mean,
            feature_std=std
        )

        # 3. Smoother & Risk Engine
        self.smoother = PredictionSmoother(
            window_size=SMOOTHING_WINDOW,
            threshold=THRESHOLD
        )
        self.risk_engine = RiskEngine()

        # 4. Scenario State Tracking
        self.fatigue_start_time: Optional[float] = None
        self.last_active_time: float = 0.0
        self.active_scenario_family: str = "NORMAL"
        self.fatigue_duration: float = 0.0
        self.frame_count: int = 0
        self.start_time: float = time.time()

    def reset(self):
        """Reset buffer riwayat dan timer kantuk."""
        self.buffer.clear()
        self.smoother.reset()
        self.fatigue_start_time = None
        self.last_active_time = 0.0
        self.active_scenario_family = "NORMAL"
        self.fatigue_duration = 0.0
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        road_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analisis satu frame gambar OpenCV BGR dari kamera/video pengemudi.
        """
        t_start = time.time()
        self.frame_count += 1
        h, w = frame_bgr.shape[:2] if frame_bgr is not None else (480, 640)

        # 1. Deteksi Landmark Wajah
        detection = self.detector.detect_face_landmarks(frame_bgr)
        landmark_valid = detection["valid"]
        landmarks = detection["landmarks"]
        bbox = detection["bbox"]

        ear_val = 0.0
        mar_val = 0.0
        left_ear = 0.0
        right_ear = 0.0
        pitch_ratio = 1.0
        yaw_ratio = 0.0
        head_direction = "CENTER"
        is_nodding = False
        is_looking_aside = False

        raw_drowsy_prob: Optional[float] = None
        smoothed_drowsy_prob: Optional[float] = None
        pred_label = "NOT DROWSY"

        status = "UNKNOWN (NO FACE)"
        scenario_label = "NO FACE"
        scenario_family = "NO_FACE"
        current_target_alert_sec = DROWSY_ALERT_SECONDS
        is_fatigue_active = False

        # Landmarks keypoints untuk rendering frontend
        key_landmarks: Dict[str, List[Dict[str, float]]] = {
            "left_eye": [],
            "right_eye": [],
            "mouth": [],
            "face_axes": []
        }

        if landmark_valid and landmarks is not None:
            # Format normalized landmarks untuk frontend canvas
            for idx in LEFT_EYE:
                if idx < len(landmarks):
                    key_landmarks["left_eye"].append({"x": landmarks[idx].x, "y": landmarks[idx].y})
            for idx in RIGHT_EYE:
                if idx < len(landmarks):
                    key_landmarks["right_eye"].append({"x": landmarks[idx].x, "y": landmarks[idx].y})
            for idx in MOUTH:
                if idx < len(landmarks):
                    key_landmarks["mouth"].append({"x": landmarks[idx].x, "y": landmarks[idx].y})
            for idx in [LM_FOREHEAD, LM_NOSE, LM_CHIN, LM_LEFT_CHEEK, LM_RIGHT_CHEEK]:
                if idx < len(landmarks):
                    key_landmarks["face_axes"].append({"x": landmarks[idx].x, "y": landmarks[idx].y})

            # 2. Ekstraksi Fitur EAR V3 & MAR
            features = extract_features(landmarks)
            if features["valid"]:
                ear_val = float(features["ear"])
                mar_val = float(features["mar"])
                left_ear = float(features.get("left_ear", 0.0))
                right_ear = float(features.get("right_ear", 0.0))

                # 3. Buffer sequence
                self.buffer.append([ear_val, mar_val])

                if self.buffer.is_ready():
                    # 4. Inferensi Model GRU
                    pred_result = self.engine.predict(self.buffer, threshold=THRESHOLD)
                    raw_drowsy_prob = float(pred_result["drowsy_probability"])

                    # 5. Temporal Smoothing
                    smooth_result = self.smoother.update(raw_drowsy_prob, threshold=THRESHOLD)
                    smoothed_drowsy_prob = float(smooth_result["smoothed_probability"])
                    pred_label = smooth_result["label"]

                    # 6. Evaluasi Head Pose
                    is_nodding, is_looking_aside, pitch_ratio, yaw_ratio, head_direction = evaluate_head_pose(
                        landmarks=landmarks,
                        nodding_ratio_threshold=NODDING_PITCH_RATIO_THRESHOLD,
                        yaw_ratio_threshold=LOOK_ASIDE_YAW_THRESHOLD,
                        ear_val=ear_val,
                        smoothed_drowsy_prob=smoothed_drowsy_prob
                    )

                    # 7. Skenario Fisiologis
                    is_eye_closed = (ear_val < EYE_CLOSED_EAR_THRESHOLD)
                    is_yawning = (mar_val >= YAWN_MAR_THRESHOLD)

                    if is_eye_closed:
                        scenario_family = "MICROSLEEP"
                        status = "MICROSLEEP (CRITICAL)"
                        scenario_label = "MICROSLEEP"
                        current_target_alert_sec = MICROSLEEP_ALERT_SECONDS
                        is_fatigue_active = True

                    elif is_nodding:
                        scenario_family = "NODDING"
                        status = "NODDING (DROWSY)"
                        scenario_label = "NODDING"
                        current_target_alert_sec = NODDING_ALERT_SECONDS
                        is_fatigue_active = True

                    elif is_looking_aside:
                        scenario_family = "LOOK_ASIDE"
                        status = f"LOOKING {head_direction} (DISTRACTED)"
                        scenario_label = f"LOOKING {head_direction}"
                        current_target_alert_sec = LOOK_ASIDE_ALERT_SECONDS
                        is_fatigue_active = True

                    elif is_yawning and not is_eye_closed and (smoothed_drowsy_prob < THRESHOLD or ear_val >= 0.23):
                        scenario_family = "YAWNING"
                        status = "YAWNING (WARNING)"
                        scenario_label = "YAWNING"
                        current_target_alert_sec = YAWN_ALERT_SECONDS
                        is_fatigue_active = True

                    elif pred_label == "DROWSY":
                        scenario_family = "DROWSY"
                        status = "DROWSY"
                        scenario_label = "DROWSY (GRU)"
                        current_target_alert_sec = DROWSY_ALERT_SECONDS
                        is_fatigue_active = True

                    else:
                        scenario_family = "NORMAL"
                        status = "NOT DROWSY"
                        scenario_label = "ALERT / NORMAL"
                        current_target_alert_sec = DROWSY_ALERT_SECONDS
                        is_fatigue_active = False

                else:
                    status = f"BUFFERING ({len(self.buffer)}/{self.sequence_length})"
                    scenario_label = "INITIALIZING"
                    scenario_family = "BUFFERING"
                    is_fatigue_active = False
            else:
                status = "UNKNOWN (INVALID FEAT)"
                scenario_label = "INVALID FEAT"
                scenario_family = "NO_FACE"
                is_fatigue_active = False
        else:
            status = "UNKNOWN (NO FACE)"
            scenario_label = "NO FACE"
            scenario_family = "NO_FACE"
            is_fatigue_active = False

        # 8. Pelacakan Durasi Alarm Kantuk
        curr_time = time.time()
        alarm_active = False

        if is_fatigue_active:
            if self.fatigue_start_time is None or self.active_scenario_family != scenario_family:
                self.fatigue_start_time = curr_time
                self.active_scenario_family = scenario_family

            self.fatigue_duration = curr_time - self.fatigue_start_time
            self.last_active_time = curr_time

            if self.fatigue_duration >= current_target_alert_sec:
                alarm_active = True
        else:
            if curr_time - self.last_active_time > 0.35:
                self.fatigue_start_time = None
                self.fatigue_duration = 0.0
                self.active_scenario_family = "NORMAL"
                alarm_active = False

        # 9. Evaluasi Risk Engine SafeRoute AI
        risk_result = self.risk_engine.evaluate_risk(
            scenario=scenario_family,
            fatigue_duration=self.fatigue_duration,
            target_alert_sec=current_target_alert_sec,
            alarm_active=alarm_active,
            smoothed_drowsy_prob=smoothed_drowsy_prob,
            is_nodding=is_nodding,
            is_looking_aside=is_looking_aside,
            head_direction=head_direction,
            ear=ear_val,
            mar=mar_val,
            road_context=road_context
        )

        latency_ms = (time.time() - t_start) * 1000.0

        # Normalisasi bounding box untuk frontend [x, y, w, h] (0.0 to 1.0)
        norm_bbox = None
        if bbox and w > 0 and h > 0:
            norm_bbox = {
                "x": bbox[0] / w,
                "y": bbox[1] / h,
                "w": bbox[2] / w,
                "h": bbox[3] / h,
                "px": bbox[0],
                "py": bbox[1],
                "pw": bbox[2],
                "ph": bbox[3]
            }

        return {
            "face_detected": landmark_valid,
            "bbox": norm_bbox,
            "key_landmarks": key_landmarks,
            "ear": round(ear_val, 4),
            "mar": round(mar_val, 4),
            "left_ear": round(left_ear, 4),
            "right_ear": round(right_ear, 4),
            "pitch_ratio": round(pitch_ratio, 3),
            "yaw_ratio": round(yaw_ratio, 3),
            "head_direction": head_direction,
            "is_nodding": is_nodding,
            "is_looking_aside": is_looking_aside,
            "raw_drowsy_prob": round(raw_drowsy_prob, 4) if raw_drowsy_prob is not None else None,
            "smoothed_drowsy_prob": round(smoothed_drowsy_prob, 4) if smoothed_drowsy_prob is not None else None,
            "status": status,
            "scenario": scenario_family,
            "scenario_label": scenario_label,
            "fatigue_duration": round(self.fatigue_duration, 2),
            "target_alert_sec": round(current_target_alert_sec, 2),
            "alarm_active": alarm_active,
            "buffer_len": len(self.buffer),
            "buffer_max": self.sequence_length,
            "latency_ms": round(latency_ms, 1),
            "risk_level": risk_result["risk_level"],
            "risk_score": risk_result["risk_score"],
            "risk_reasons": risk_result["risk_reasons"],
            "alert_message": risk_result["alert_message"]
        }
