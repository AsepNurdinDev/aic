import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
# PROCESSOR
# ==============================================================================
class DrowsinessProcessor:
    def __init__(self, model_path: str, landmarker_path: str, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DrowsinessModel().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.mean = np.array(checkpoint['normalization_mean'], dtype=np.float32)
        self.std = np.array(checkpoint['normalization_std'], dtype=np.float32)
        
        if len(self.mean) != 36 or len(self.std) != 36:
            raise ValueError(f"Normalization stats length mismatch! Expected 36, got {len(self.mean)}.")
            
        self.sequence_length = 16
        self.base_buffer = deque(maxlen=self.sequence_length)
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        base_options = python.BaseOptions(model_asset_path=landmarker_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]
        self.mouth = [61, 291, 13, 14]
        self.nose = 1
        
        self.last_result = None

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def calculate_ear(self, points):
        vertical_1 = self.calculate_distance(points[1], points[5])
        vertical_2 = self.calculate_distance(points[2], points[4])
        horizontal = self.calculate_distance(points[0], points[3])
        if horizontal == 0:
            return 0
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
        if eye_distance < 1e-6:
            return None
            
        mouth_width_dist = self.calculate_distance(pt_61, pt_291)
        mouth_height_dist = self.calculate_distance(pt_13, pt_14)
        mouth_width = mouth_width_dist / eye_distance
        mouth_height = mouth_height_dist / eye_distance
        
        safe_mouth_width = max(mouth_width, 1e-6)
        mar = mouth_height / safe_mouth_width
        
        face_center = (pt_33 + pt_263) / 2.0
        face_center_x = (pt_1[0] - face_center[0]) / eye_distance
        face_center_y = (pt_1[1] - face_center[1]) / eye_distance
        
        eye_center = (pt_33 + pt_263) / 2.0
        head_x = (pt_1[0] - eye_center[0]) / eye_distance
        head_y = (pt_1[1] - eye_center[1]) / eye_distance
        
        eye_vector = pt_263 - pt_33
        head_roll = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
        if head_roll > 90:
            head_roll -= 180
        elif head_roll < -90:
            head_roll += 180
            
        base_features = np.array([
            ear_left, ear_right, ear_mean, mar, mouth_width, mouth_height,
            eye_distance, face_center_x, face_center_y, head_x, head_y, head_roll
        ], dtype=np.float32)
        
        if np.isnan(base_features).any() or np.isinf(base_features).any():
            return None
            
        return base_features

    def process(self, frame_rgb: np.ndarray) -> dict:
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.face_landmarker.detect(mp_image)
            
            if not results.face_landmarks:
                self.base_buffer.clear()
                self.feature_buffer.clear()
                self.last_result = None
                return {"drowsiness_available": 0}
                
            h, w, _ = frame_rgb.shape
            face_landmarks = results.face_landmarks[0]
            
            base_features = self.extract_base_features(face_landmarks, w, h)
            if base_features is None:
                self.last_result = None
                return {"drowsiness_available": 0}
                
            self.base_buffer.append(base_features)
            
            if len(self.base_buffer) >= 1:
                base_t = self.base_buffer[-1]
                if len(self.base_buffer) >= 2:
                    base_t_minus_1 = self.base_buffer[-2]
                    delta_t = base_t - base_t_minus_1
                else:
                    delta_t = np.zeros_like(base_t)
                    
                if len(self.base_buffer) >= 3:
                    base_t_minus_2 = self.base_buffer[-3]
                    delta_t_minus_1 = base_t_minus_1 - base_t_minus_2
                    delta2_t = delta_t - delta_t_minus_1
                else:
                    delta2_t = np.zeros_like(base_t)
                    
                final_vector = np.concatenate([base_t, delta_t, delta2_t])
                self.feature_buffer.append(final_vector)
                
            if len(self.feature_buffer) < self.sequence_length:
                self.last_result = None
                return {"drowsiness_available": 0}
                
            features = np.array(self.feature_buffer)
            features = (features - self.mean) / self.std
            
            if np.isnan(features).any() or np.isinf(features).any():
                self.last_result = None
                return {"drowsiness_available": 0}
                
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                bin_out, class_out = self.model(x)
                bin_probs = torch.softmax(bin_out, dim=1)[0].cpu().numpy()
                class_probs = torch.softmax(class_out, dim=1)[0].cpu().numpy()
                
            states = ["alert", "microsleep", "yawning"]
            state_idx = np.argmax(class_probs)
            
            self.last_result = {
                "drowsiness_available": 1,
                "not_drowsy_prob": float(bin_probs[0]),
                "drowsy_prob": float(bin_probs[1]),
                "alert_prob": float(class_probs[0]),
                "microsleep_prob": float(class_probs[1]),
                "yawning_prob": float(class_probs[2]),
                "prediction_confidence": float(max(bin_probs)),
                "state": states[state_idx]
            }
            return self.last_result
            
        except Exception as e:
            self.last_result = None
            return {"drowsiness_available": 0}
