import cv2
import math
import time
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# CONFIG
# ==============================================================================
DROWSY_THRESHOLD = 0.70
INFERENCE_HZ = 5.0
DROWSY_ALARM_DURATION = 2.0
YAWNING_ALARM_DURATION = 5.0

# ==============================================================================
# MODEL ARCHITECTURE (V5 - 36 Features)
# ==============================================================================
class DrowsinessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(36, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30)
        )
        self.gru = nn.GRU(64, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.30)
        
        self.attention = nn.ModuleDict({
            'score': nn.Sequential(
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
        })
        
        self.shared = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.30)
        )
        
        self.binary_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 2)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        
        attn_weights = self.attention['score'](x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = torch.sum(x * attn_weights, dim=1)
        
        x = self.shared(x)
        bin_out = self.binary_head(x)
        class_out = self.class_head(x)
        return bin_out, class_out

# ==============================================================================
# DRIVER ANALYZER CORE
# ==============================================================================
class DriverAnalyzer:
    def __init__(self, checkpoint_path, config_path=None, mediapipe_model_path=None, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DrowsinessModel().to(self.device)
        
        # Load Model
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Extrack Norm stats
        self.mean = np.array(checkpoint['normalization_mean'], dtype=np.float32)
        self.std = np.array(checkpoint['normalization_std'], dtype=np.float32)
        
        # Buffer
        self.sequence_length = 16
        self.base_buffer = deque(maxlen=self.sequence_length)
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        # MediaPipe
        base_options = python.BaseOptions(model_asset_path=mediapipe_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Landmarks
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]
        self.mouth = [61, 291, 13, 14]
        self.nose = 1
        
        # State tracking
        self.inference_interval = 1.0 / INFERENCE_HZ
        self.last_inference_time = 0.0
        self.last_prob = 0.0
        self.last_state = "-"
        self.last_status = "WAITING"
        
        # Durations
        self.drowsy_start_time = None
        self.drowsy_alarm_active = False
        self.yawning_start_time = None
        self.yawning_alarm_active = False
        
    def reset(self):
        self.base_buffer.clear()
        self.feature_buffer.clear()
        self.last_inference_time = 0.0
        self.last_prob = 0.0
        self.last_state = "-"
        self.last_status = "WAITING"
        self.drowsy_start_time = None
        self.drowsy_alarm_active = False
        self.yawning_start_time = None
        self.yawning_alarm_active = False

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def calculate_ear(self, points):
        vertical_1 = self.calculate_distance(points[1], points[5])
        vertical_2 = self.calculate_distance(points[2], points[4])
        horizontal = self.calculate_distance(points[0], points[3])
        if horizontal == 0: return 0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def extract_base_features(self, landmarks, w, h):
        def get_pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
            
        le_pts = [get_pt(i) for i in self.left_eye]
        re_pts = [get_pt(i) for i in self.right_eye]
        
        pt_33 = get_pt(33)
        pt_263 = get_pt(263)
        pt_1 = get_pt(1)
        pt_61 = get_pt(61)
        pt_291 = get_pt(291)
        pt_13 = get_pt(13)
        pt_14 = get_pt(14)
        
        ear_left = self.calculate_ear(le_pts)
        ear_right = self.calculate_ear(re_pts)
        ear_mean = (ear_left + ear_right) / 2.0
        
        eye_distance = self.calculate_distance(pt_33, pt_263)
        if eye_distance < 1e-6: return None
            
        mouth_width_dist = self.calculate_distance(pt_61, pt_291)
        mouth_height_dist = self.calculate_distance(pt_13, pt_14)
        mouth_width = mouth_width_dist / eye_distance
        mouth_height = mouth_height_dist / eye_distance
        
        safe_mouth_width = max(mouth_width, 1e-6)
        mar = mouth_height / safe_mouth_width
        
        face_center = (pt_33 + pt_263) / 2.0
        face_center_x = (pt_1[0] - face_center[0]) / eye_distance
        face_center_y = (pt_1[1] - face_center[1]) / eye_distance
        
        eye_vector = pt_263 - pt_33
        eye_center = (pt_33 + pt_263) / 2.0
        head_x = (pt_1[0] - eye_center[0]) / eye_distance
        head_y = (pt_1[1] - eye_center[1]) / eye_distance
        
        head_roll = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
        if head_roll > 90: head_roll -= 180
        elif head_roll < -90: head_roll += 180
            
        base_features = np.array([
            ear_left, ear_right, ear_mean, mar, mouth_width, mouth_height,
            eye_distance, face_center_x, face_center_y, head_x, head_y, head_roll
        ], dtype=np.float32)
        
        if np.isnan(base_features).any() or np.isinf(base_features).any():
            return None
        return base_features, ear_mean, mar

    def process_frame(self, frame_bgr: np.ndarray, road_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        timestamp_ms = int(time.monotonic() * 1000)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        h, w, _ = frame_bgr.shape
        
        if not results.face_landmarks:
            self.drowsy_start_time = None
            self.yawning_start_time = None
            self.drowsy_alarm_active = False
            self.yawning_alarm_active = False
            return self._build_response("UNKNOWN", "NO FACE", 0, 0, 0.0, 0.0)
            
        face_landmarks = results.face_landmarks[0]
        
        extracted = self.extract_base_features(face_landmarks, w, h)
        if extracted is None:
            return self._build_response("UNKNOWN", "INVALID FEATURES", 0, 0, 0.0, 0.0)
            
        base_features, ear_val, mar_val = extracted
        self.base_buffer.append(base_features)
        
        if len(self.base_buffer) >= 1:
            base_t = self.base_buffer[-1]
            delta_t = base_t - self.base_buffer[-2] if len(self.base_buffer) >= 2 else np.zeros_like(base_t)
            delta2_t = delta_t - (self.base_buffer[-2] - self.base_buffer[-3]) if len(self.base_buffer) >= 3 else np.zeros_like(base_t)
            
            final_vector = np.concatenate([base_t, delta_t, delta2_t])
            self.feature_buffer.append(final_vector)
            
        current_time = time.monotonic()
        
        if len(self.feature_buffer) == self.sequence_length:
            if current_time - self.last_inference_time >= self.inference_interval:
                self.last_inference_time = current_time
                
                features = np.array(self.feature_buffer)
                features = (features - self.mean) / self.std
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    bin_out, class_out = self.model(x)
                    bin_probs = torch.softmax(bin_out, dim=1)[0]
                    state_idx = torch.argmax(class_out, dim=1).item()
                    states = ["ALERT", "MICROSLEEP", "YAWNING"]
                    
                    self.last_prob = bin_probs[1].item()
                    self.last_state = states[state_idx]
                    
                    if self.last_prob >= DROWSY_THRESHOLD:
                        self.last_status = "DROWSY"
                    else:
                        self.last_status = "NOT DROWSY"

        # Update Alarm Policy
        if self.last_status == "DROWSY":
            if self.drowsy_start_time is None: self.drowsy_start_time = current_time
            if (current_time - self.drowsy_start_time) >= DROWSY_ALARM_DURATION:
                self.drowsy_alarm_active = True
        else:
            self.drowsy_start_time = None
            self.drowsy_alarm_active = False
            
        if self.last_state == "YAWNING":
            if self.yawning_start_time is None: self.yawning_start_time = current_time
            if (current_time - self.yawning_start_time) >= YAWNING_ALARM_DURATION:
                self.yawning_alarm_active = True
        else:
            self.yawning_start_time = None
            self.yawning_alarm_active = False

        alarm_active = self.drowsy_alarm_active or self.yawning_alarm_active
        risk_level = "CRITICAL" if self.drowsy_alarm_active else ("WARNING" if self.yawning_alarm_active else "LOW")
        risk_score = 100 if self.drowsy_alarm_active else (75 if self.yawning_alarm_active else int(self.last_prob * 100))
        
        return self._build_response(self.last_status, self.last_state, risk_level, risk_score, ear_val, mar_val, alarm_active)

    def _build_response(self, status, scenario, risk_level, risk_score, ear, mar, alarm=False):
        return {
            "status": status,
            "scenario": scenario,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "alarm_active": alarm,
            "raw_prob": round(self.last_prob, 3)
        }
