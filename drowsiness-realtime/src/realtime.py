"""
src/realtime.py

Aplikasi Deteksi Kantuk Pengemudi Real-Time via Webcam menggunakan:
- MediaPipe Tasks Face Landmarker
- Ekstraktor Fitur EAR V3 + MAR + Head Pose (Nodding & Look-Aside)
- Sliding Sequence Buffer (60 frame) + Training Normalization
- Model LandmarkGRU (best_landmark_gru_ear_mar.pth)
- Temporal Prediction Smoother
- Audio Alert System & Skenario Kantuk Realistis:
    1. Microsleep / Mata Terpejam Kritis -> Respon cepat alarm (>= 1.0s)
    2. Head Nodding / Kepala Mengangguk-Terkulai -> Respon alarm (>= 1.5s)
    3. Looking Aside / Menengok Kiri-Kanan -> Respon distraksi alarm (>= 3.0s)
    4. Drowsy / Pola Kantuk Umum Model GRU -> Alarm standar (>= 4.0s)
    5. Yawning / Menguap Saja -> Peringatan awal toleran (>= 4.5s)
    6. Alert / Sadar -> Reset kondisi

Penggunaan:
    python -m src.realtime
    python -m src.realtime --camera-index 0
    python -m src.realtime --threshold 0.50
    python -m src.realtime --drowsy-alert-sec 4.0
    python -m src.realtime --nodding-alert-sec 1.5
    python -m src.realtime --look-aside-alert-sec 3.0
    python -m src.realtime --log
"""

import os
import sys
import time
import math
import threading
import argparse
import csv
from datetime import datetime
from typing import Optional, Any, Tuple
import cv2
import numpy as np
import torch

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

from src.config import (
    MODEL_PATH,
    CONFIG_PATH,
    MEDIAPIPE_MODEL_PATH,
    DEFAULT_LOG_PATH,
    SEQUENCE_LENGTH,
    THRESHOLD,
    SMOOTHING_WINDOW,
    CAMERA_INDEX,
    DROWSY_ALERT_SECONDS,
    NODDING_ALERT_SECONDS,
    LOOK_ASIDE_ALERT_SECONDS,
    MICROSLEEP_ALERT_SECONDS,
    YAWN_ALERT_SECONDS,
    EYE_CLOSED_EAR_THRESHOLD,
    YAWN_MAR_THRESHOLD,
    NODDING_PITCH_RATIO_THRESHOLD,
    LOOK_ASIDE_YAW_THRESHOLD,
    ALARM_BEEP_FREQUENCY,
    ALARM_BEEP_DURATION_MS,
    ENABLE_AUDIO_ALERT,
    CLASS_NAMES,
    COLOR_NOT_DROWSY,
    COLOR_DROWSY,
    COLOR_NODDING,
    COLOR_LOOK_ASIDE,
    COLOR_YAWNING,
    COLOR_MICROSLEEP,
    COLOR_BUFFERING,
    COLOR_UNKNOWN,
    COLOR_TEXT,
    COLOR_OVERLAY_BG,
)
from src.landmark_detector import FaceLandmarkDetector
from src.feature_extractor import extract_features, LEFT_EYE, RIGHT_EYE, MOUTH
from src.preprocessing import SequenceBuffer, load_training_statistics
from src.inference import DrowsinessInference
from src.smoothing import PredictionSmoother


# Landmark indices untuk Head Pose (Nodding & Menengok Kiri/Kanan)
LM_FOREHEAD = 10
LM_NOSE = 1
LM_CHIN = 152
LM_LEFT_CHEEK = 234
LM_RIGHT_CHEEK = 454
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263


def evaluate_head_pose(
    landmarks: Any,
    nodding_ratio_threshold: float = NODDING_PITCH_RATIO_THRESHOLD,
    yaw_ratio_threshold: float = LOOK_ASIDE_YAW_THRESHOLD,
    ear_val: float = 0.30,
    smoothed_drowsy_prob: float = 0.0
) -> Tuple[bool, bool, float, float, str]:
    """
    Menganalisis orientasi kepala pengemudi (Nodding & Looking Aside):
    - Head Nodding (Menunduk/Mengangguk):
      Dikalibrasi agar tidak agresif terhadap gerakan membaca biasa:
      1. Menunduk dalam (pitch_ratio < 0.45), ATAU
      2. Menunduk sedang (pitch_ratio < 0.52) disertai tanda mata kantuk (ear < 0.25 / smoothed_prob > 0.40).
    - Looking Aside (Menengok Kiri / Kanan):
      Terdeteksi jika |yaw_ratio| >= yaw_ratio_threshold (default 0.28 untuk putaran kepala ~20 derajat)
      atau rotasi 3D sudut yaw >= 18.0 derajat.

    Returns:
        is_nodding: bool
        is_looking_aside: bool
        pitch_ratio: float
        yaw_ratio: float
        direction: str ("CENTER", "LEFT", "RIGHT", "DOWN")
    """
    if landmarks is None or len(landmarks) < 455:
        return False, False, 1.0, 0.0, "CENTER"

    try:
        p_forehead = landmarks[LM_FOREHEAD]
        p_nose = landmarks[LM_NOSE]
        p_chin = landmarks[LM_CHIN]
        p_left_cheek = landmarks[LM_LEFT_CHEEK]
        p_right_cheek = landmarks[LM_RIGHT_CHEEK]
        p_left_eye = landmarks[LM_LEFT_EYE_OUTER]
        p_right_eye = landmarks[LM_RIGHT_EYE_OUTER]

        # 1. Rasio Pitch Vertikal (Dahi-Hidung-Dagu)
        upper_dist = abs(p_nose.y - p_forehead.y)
        lower_dist = abs(p_chin.y - p_nose.y)
        pitch_ratio = lower_dist / max(1e-6, upper_dist)

        # 2. Rasio Yaw Horizontal 2D (Pipi Kiri - Hidung - Pipi Kanan)
        dist_l = abs(p_nose.x - p_left_cheek.x)
        dist_r = abs(p_right_cheek.x - p_nose.x)
        total_w = dist_l + dist_r
        # Nilai negatif = Menengok Kiri; Nilai positif = Menengok Kanan
        yaw_ratio = (dist_l - dist_r) / max(1e-6, total_w)

        # 3. Sudut Yaw 3D (Perbedaan kedalaman mata kiri vs mata kanan)
        yaw_deg = math.degrees(math.atan2(p_left_eye.z - p_right_eye.z, max(1e-6, abs(p_right_eye.x - p_left_eye.x))))

        # Evaluasi Nodding (Tidak agresif terhadap posisi mengetik/laptop biasa)
        is_deep_nod = (pitch_ratio < nodding_ratio_threshold)
        is_drowsy_nod = (pitch_ratio < (nodding_ratio_threshold + 0.07)) and (ear_val < 0.25 or smoothed_drowsy_prob > 0.40)
        is_nodding = bool(is_deep_nod or is_drowsy_nod)

        # Evaluasi Menengok Kiri/Kanan (2D ratio atau 3D yaw)
        is_looking_aside = bool((abs(yaw_ratio) >= yaw_ratio_threshold) or (abs(yaw_deg) >= 18.0))

        # Tentukan arah hadap
        direction = "CENTER"
        if is_looking_aside:
            direction = "LEFT" if (yaw_ratio < 0 or yaw_deg > 0) else "RIGHT"
        elif is_nodding:
            direction = "DOWN"

        return is_nodding, is_looking_aside, float(pitch_ratio), float(yaw_ratio), direction

    except Exception:
        return False, False, 1.0, 0.0, "CENTER"


class AudioAlertManager:
    """
    Manager suara alarm kantuk non-blocking yang berjalan di background thread
    agar tidak mengganggu frame rate inferensi video realtime.
    """

    def __init__(
        self,
        frequency: int = ALARM_BEEP_FREQUENCY,
        duration_ms: int = ALARM_BEEP_DURATION_MS,
        enabled: bool = True
    ):
        self.frequency = frequency
        self.duration_ms = duration_ms
        self.enabled = enabled
        self._is_playing = False
        self._lock = threading.Lock()

    def trigger_alarm(self):
        """Memicu bunyi beep non-blocking jika belum ada bunyi yang sedang berjalan."""
        if not self.enabled:
            return

        with self._lock:
            if self._is_playing:
                return
            self._is_playing = True

        def _play():
            try:
                if HAS_WINSOUND:
                    winsound.Beep(self.frequency, self.duration_ms)
                else:
                    # Fallback untuk lingkungan non-Windows
                    print("\a", end="", flush=True)
            except Exception:
                pass
            finally:
                with self._lock:
                    self._is_playing = False

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()


def perform_startup_check(
    checkpoint_path: str,
    config_path: str,
    device: str
) -> DrowsinessInference:
    """
    Menjalankan startup check sebelum membuka webcam.
    """
    print("=" * 65)
    print("           REALTIME DROWSINESS DETECTION - STARTUP CHECK        ")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        print(f"[FATAL ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    try:
        # Inisialisasi Inference Engine
        engine = DrowsinessInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            default_threshold=THRESHOLD
        )

        # Hitung parameter model
        total_params = sum(p.numel() for p in engine.model.parameters())
        trainable_params = sum(p.numel() for p in engine.model.parameters() if p.requires_grad)

        # Jalankan dummy forward pass (1, 60, 2)
        dummy_input = torch.randn(1, SEQUENCE_LENGTH, 2).to(engine.device)
        dummy_output = engine.predict(dummy_input)

        print(f"[*] Target Device        : {engine.device}")
        print(f"[*] Checkpoint Path      : {checkpoint_path}")
        print(f"[*] Config Path          : {config_path}")
        print(f"[*] Total Parameters     : {total_params:,} (Trainable: {trainable_params:,})")
        print(f"[*] Model Input Shape    : (1, {SEQUENCE_LENGTH}, 2) [EAR, MAR]")
        print(f"[*] Output Classes       : {CLASS_NAMES}")
        print(f"[*] Dummy Test Output    : {dummy_output}")
        print("=" * 65)
        print("[SUCCESS] Model initialized & verified. Ready for webcam stream.")
        print("=" * 65)
        return engine

    except Exception as e:
        print(f"\n[FATAL ERROR] Startup check failed: {e}")
        print("Inference model could not be initialized from checkpoint. Aborting.")
        sys.exit(1)


def draw_visuals(
    frame: np.ndarray,
    landmarks: Optional[Any],
    bbox: Optional[tuple],
    status_color: tuple,
    is_nodding: bool = False,
    is_looking_aside: bool = False,
    head_direction: str = "CENTER"
):
    """
    Gambar bounding box dan titik landmark mata serta mulut pada frame.
    """
    if frame is None or landmarks is None:
        return

    h, w = frame.shape[:2]

    # Gambar bounding box jika tersedia
    if bbox:
        x, y, bw, bh = bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), status_color, 2)
        if is_nodding:
            cv2.putText(frame, "NODDING DETECTED", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif is_looking_aside:
            cv2.putText(frame, f"LOOKING {head_direction}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    # Highlight Eye Landmarks (Kiri & Kanan)
    for idx in LEFT_EYE + RIGHT_EYE:
        if idx < len(landmarks):
            lm = landmarks[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

    # Highlight Mouth Landmarks
    for idx in MOUTH:
        if idx < len(landmarks):
            lm = landmarks[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)

    # Highlight Head Pose Reference points
    for idx in [LM_FOREHEAD, LM_NOSE, LM_CHIN, LM_LEFT_CHEEK, LM_RIGHT_CHEEK]:
        if idx < len(landmarks):
            lm = landmarks[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 4, (0, 165, 255), -1)


def draw_dashboard_overlay(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    ear: float,
    mar: float,
    pitch_ratio: float,
    yaw_ratio: float,
    head_direction: str,
    raw_drowsy_prob: Optional[float],
    smoothed_drowsy_prob: Optional[float],
    status: str,
    scenario_label: str,
    status_color: tuple,
    landmark_valid: bool,
    buffer_len: int,
    buffer_max: int,
    fatigue_duration: float = 0.0,
    target_alert_sec: float = DROWSY_ALERT_SECONDS,
    alarm_active: bool = False
):
    """
    Gambar dashboard status informasi transparan di pojok kiri atas frame
    serta banner peringatan suara ketika terdeteksi kantuk lebih dari batas waktu.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    box_w, box_h = 360, 310
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), COLOR_OVERLAY_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), status_color, 2)

    # Header Status
    cv2.putText(frame, f"STATUS: {status}", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.65, status_color, 2)

    # Indikator Landmark & Buffer
    lm_text = "VALID" if landmark_valid else "INVALID"
    lm_color = COLOR_NOT_DROWSY if landmark_valid else COLOR_DROWSY
    cv2.putText(frame, f"Landmark: {lm_text}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lm_color, 1)
    cv2.putText(frame, f"Buffer  : {buffer_len}/{buffer_max}", (190, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    # Metrik EAR / MAR
    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    cv2.putText(frame, f"MAR: {mar:.3f}", (190, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)

    # Metrik Head Pose (Pitch & Yaw)
    pose_str = f"Head: {head_direction} (P:{pitch_ratio:.2f}, Y:{yaw_ratio:+.2f})"
    cv2.putText(frame, pose_str, (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

    # Probabilitas Model GRU
    raw_str = f"{raw_drowsy_prob:.2f}" if raw_drowsy_prob is not None else "--"
    sm_str = f"{smoothed_drowsy_prob:.2f}" if smoothed_drowsy_prob is not None else "--"
    cv2.putText(frame, f"Raw Prob     : {raw_str}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    cv2.putText(frame, f"Smoothed Prob: {sm_str}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    # Scenario Indicator
    cv2.putText(frame, f"Scenario     : {scenario_label}", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # Duration Tracker
    if fatigue_duration > 0:
        time_color = COLOR_DROWSY if alarm_active else COLOR_BUFFERING
        cv2.putText(
            frame,
            f"Alert Timer  : {fatigue_duration:.1f}s / {target_alert_sec:.1f}s",
            (20, 222),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            time_color,
            1
        )
    else:
        cv2.putText(
            frame,
            f"Alert Timer  : 0.0s / {target_alert_sec:.1f}s",
            (20, 222),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1
        )

    # Performance
    cv2.putText(frame, f"FPS    : {fps:.1f}", (20, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (190, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Petunjuk Exit
    cv2.putText(frame, "Press 'q' to Quit", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Peringatan Visual Tambahan jika Alarm Aktif
    if alarm_active:
        banner_overlay = frame.copy()
        banner_h = 54
        cv2.rectangle(banner_overlay, (0, h - banner_h), (w, h), (0, 0, 220), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
        banner_text = f"ALARM: {scenario_label} > {target_alert_sec:.1f} DETIK!"
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
        text_x = max(10, (w - text_size[0]) // 2)
        cv2.putText(frame, banner_text, (text_x, h - 17), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)


def run_realtime(
    camera_index: int = CAMERA_INDEX,
    threshold: float = THRESHOLD,
    smoothing_window: int = SMOOTHING_WINDOW,
    drowsy_alert_sec: float = DROWSY_ALERT_SECONDS,
    nodding_alert_sec: float = NODDING_ALERT_SECONDS,
    look_aside_alert_sec: float = LOOK_ASIDE_ALERT_SECONDS,
    microsleep_alert_sec: float = MICROSLEEP_ALERT_SECONDS,
    yawn_alert_sec: float = YAWN_ALERT_SECONDS,
    eye_thresh: float = EYE_CLOSED_EAR_THRESHOLD,
    yawn_thresh: float = YAWN_MAR_THRESHOLD,
    nodding_ratio_thresh: float = NODDING_PITCH_RATIO_THRESHOLD,
    look_aside_yaw_thresh: float = LOOK_ASIDE_YAW_THRESHOLD,
    enable_sound: bool = ENABLE_AUDIO_ALERT,
    device: str = "cpu",
    log_file: Optional[str] = None,
    no_display: bool = False
):
    """
    Main loop untuk deteksi kantuk real-time via webcam dengan skenario realistis:
    - General Drowsiness Alert : 4.0 detik
    - Nodding Alert            : 2.0 detik
    - Looking Aside Alert      : 3.0 detik
    - Microsleep Alert         : 1.0 detik
    - Yawning Alert            : 4.5 detik
    """
    # 1. Startup Check
    engine = perform_startup_check(
        checkpoint_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        device=device
    )

    # 2. Inisialisasi Detektor Landmark MediaPipe
    print(f"[*] Initializing MediaPipe Face Landmarker from: {MEDIAPIPE_MODEL_PATH}")
    detector = FaceLandmarkDetector(model_path=MEDIAPIPE_MODEL_PATH)

    # 3. Inisialisasi Preprocessing SequenceBuffer, Smoother, & Audio Alert
    mean, std, seq_len = load_training_statistics(
        checkpoint_path=MODEL_PATH,
        config_path=CONFIG_PATH
    )
    buffer = SequenceBuffer(
        sequence_length=seq_len,
        feature_mean=mean,
        feature_std=std
    )
    smoother = PredictionSmoother(
        window_size=smoothing_window,
        threshold=threshold
    )
    audio_alert = AudioAlertManager(
        frequency=ALARM_BEEP_FREQUENCY,
        duration_ms=ALARM_BEEP_DURATION_MS,
        enabled=enable_sound
    )

    print(f"[*] Audio Alert Status   : {'ENABLED' if enable_sound else 'DISABLED'}")
    print(f"[*] Alert Thresholds     : General Drowsy >= {drowsy_alert_sec:.1f}s | Nodding >= {nodding_alert_sec:.1f}s | Look-Aside >= {look_aside_alert_sec:.1f}s | Microsleep >= {microsleep_alert_sec:.1f}s")
    print(f"[*] Feature Thresholds   : Eye Closed EAR < {eye_thresh:.2f} | Yawning MAR >= {yawn_thresh:.2f} | Nodding Ratio < {nodding_ratio_thresh:.2f} | Look-Aside Yaw >= {look_aside_yaw_thresh:.2f}")

    # 4. Inisialisasi Webcam
    print(f"[*] Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[FATAL ERROR] Could not open webcam at index {camera_index}.")
        print("Please check camera connection or specify another index with --camera-index.")
        return

    # 5. Inisialisasi Logger jika diaktifkan
    csv_writer = None
    csv_file_obj = None
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        csv_file_obj = open(log_file, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file_obj)
        csv_writer.writerow([
            "timestamp", "frame_idx", "fps", "latency_ms",
            "landmark_valid", "ear", "mar", "pitch_ratio", "yaw_ratio",
            "raw_drowsy_prob", "smoothed_drowsy_prob", "status",
            "scenario", "fatigue_duration_sec", "alarm_active"
        ])
        print(f"[*] Logging predictions to: {log_file}")

    print("\n" + "=" * 65)
    print(" REAL-TIME DROWSINESS DETECTION RUNNING - PRESS 'q' TO STOP")
    print("=" * 65 + "\n")

    # Statistik runtime & scenario tracker
    frame_count = 0
    valid_frames = 0
    invalid_frames = 0
    start_time = time.time()
    fps = 0.0

    fatigue_start_time: Optional[float] = None
    last_active_time: float = 0.0
    active_scenario_family: str = "NORMAL"
    fatigue_duration: float = 0.0
    current_target_alert_sec: float = drowsy_alert_sec
    alarm_active: bool = False

    try:
        while True:
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Failed to grab frame from camera.")
                time.sleep(0.01)
                continue

            frame_count += 1

            # 1. Deteksi Landmark Wajah
            t_infer_start = time.time()
            detection = detector.detect_face_landmarks(frame)
            landmark_valid = detection["valid"]
            landmarks = detection["landmarks"]
            bbox = detection["bbox"]

            raw_drowsy_prob: Optional[float] = None
            smoothed_drowsy_prob: Optional[float] = None
            ear_val: float = 0.0
            mar_val: float = 0.0
            pitch_ratio: float = 1.0
            yaw_ratio: float = 0.0
            head_direction: str = "CENTER"
            is_nodding: bool = False
            is_looking_aside: bool = False
            status: str = "UNKNOWN"
            scenario_label: str = "UNKNOWN"
            scenario_family: str = "UNKNOWN"
            status_color = COLOR_UNKNOWN
            is_fatigue_active = False

            if landmark_valid and landmarks is not None:
                valid_frames += 1

                # 2. Ekstraksi Fitur EAR V3 + MAR
                features = extract_features(landmarks)

                if features["valid"]:
                    ear_val = float(features["ear"])
                    mar_val = float(features["mar"])

                    # 3. Masukkan ke Sequence Buffer
                    buffer.append([ear_val, mar_val])

                    # 4. Cek apakah buffer sudah mencukupi 60 frame
                    if buffer.is_ready():
                        # 5. Inferensi Model LandmarkGRU
                        pred_result = engine.predict(buffer, threshold=threshold)
                        raw_drowsy_prob = pred_result["drowsy_probability"]

                        # 6. Temporal Smoothing
                        smooth_result = smoother.update(raw_drowsy_prob, threshold=threshold)
                        smoothed_drowsy_prob = smooth_result["smoothed_probability"]
                        pred_label = smooth_result["label"]

                        # 7. Evaluasi Orientasi Kepala (Nodding & Looking Aside)
                        is_nodding, is_looking_aside, pitch_ratio, yaw_ratio, head_direction = evaluate_head_pose(
                            landmarks=landmarks,
                            nodding_ratio_threshold=nodding_ratio_thresh,
                            yaw_ratio_threshold=look_aside_yaw_thresh,
                            ear_val=ear_val,
                            smoothed_drowsy_prob=smoothed_drowsy_prob
                        )

                        # 8. Evaluasi Skenario Realistis Berdasarkan Prioritas:
                        is_eye_closed = (ear_val < eye_thresh)
                        is_yawning = (mar_val >= yawn_thresh)

                        if is_eye_closed:
                            # Skenario 1: Microsleep / Mata Terpejam (Kritis -> 1.0 detik)
                            scenario_family = "MICROSLEEP"
                            status = "MICROSLEEP (CRITICAL)"
                            scenario_label = "MICROSLEEP"
                            status_color = COLOR_MICROSLEEP
                            current_target_alert_sec = microsleep_alert_sec
                            is_fatigue_active = True

                        elif is_nodding:
                            # Skenario 2: Head Nodding / Mengangguk-Terkulai (Nodding -> 2.0 detik)
                            scenario_family = "NODDING"
                            status = "NODDING (DROWSY)"
                            scenario_label = "NODDING"
                            status_color = COLOR_NODDING
                            current_target_alert_sec = nodding_alert_sec
                            is_fatigue_active = True

                        elif is_looking_aside:
                            # Skenario 3: Menengok Kiri/Kanan (Distraksi -> 3.0 detik)
                            scenario_family = "LOOK_ASIDE"
                            status = f"LOOKING {head_direction} (DISTRACTED)"
                            scenario_label = f"LOOKING {head_direction}"
                            status_color = COLOR_LOOK_ASIDE
                            current_target_alert_sec = look_aside_alert_sec
                            is_fatigue_active = True

                        elif is_yawning and not is_eye_closed and (smoothed_drowsy_prob < threshold or ear_val >= 0.23):
                            # Skenario 4: Menguap Saja (Peringatan Dini Toleran -> 4.5 detik)
                            scenario_family = "YAWNING"
                            status = "YAWNING (WARNING)"
                            scenario_label = "YAWNING"
                            status_color = COLOR_YAWNING
                            current_target_alert_sec = yawn_alert_sec
                            is_fatigue_active = True

                        elif pred_label == "DROWSY":
                            # Skenario 5: Pola Kantuk Umum Model GRU (Kantuk Umum -> 4.0 detik)
                            scenario_family = "DROWSY"
                            status = "DROWSY"
                            scenario_label = "DROWSY (GRU)"
                            status_color = COLOR_DROWSY
                            current_target_alert_sec = drowsy_alert_sec
                            is_fatigue_active = True

                        else:
                            # Skenario 6: Pengemudi Sadar / Waspada (Normal)
                            scenario_family = "NORMAL"
                            status = "NOT DROWSY"
                            scenario_label = "ALERT / NORMAL"
                            status_color = COLOR_NOT_DROWSY
                            current_target_alert_sec = drowsy_alert_sec
                            is_fatigue_active = False

                    else:
                        status = f"BUFFERING ({len(buffer)}/{seq_len})"
                        scenario_label = "INITIALIZING"
                        scenario_family = "INITIALIZING"
                        status_color = COLOR_BUFFERING
                        is_fatigue_active = False
                else:
                    status = "UNKNOWN (INVALID FEAT)"
                    scenario_label = "INVALID FEAT"
                    scenario_family = "UNKNOWN"
                    status_color = COLOR_UNKNOWN
                    is_fatigue_active = False
            else:
                invalid_frames += 1
                status = "UNKNOWN (NO FACE)"
                scenario_label = "NO FACE"
                scenario_family = "UNKNOWN"
                status_color = COLOR_UNKNOWN
                is_fatigue_active = False

            # 9. Pelacakan Durasi & Pemicu Alarm Audio Sesuai Skenario
            curr_time = time.time()
            if is_fatigue_active:
                if fatigue_start_time is None or active_scenario_family != scenario_family:
                    fatigue_start_time = curr_time
                    active_scenario_family = scenario_family

                fatigue_duration = curr_time - fatigue_start_time
                last_active_time = curr_time

                if fatigue_duration >= current_target_alert_sec:
                    alarm_active = True
                    audio_alert.trigger_alarm()
                else:
                    alarm_active = False
            else:
                # Toleransi glitch / single frame drop (0.35s grace period)
                if curr_time - last_active_time > 0.35:
                    fatigue_start_time = None
                    fatigue_duration = 0.0
                    active_scenario_family = "NORMAL"
                    alarm_active = False

            latency_ms = (curr_time - t_infer_start) * 1000.0

            # Hitung FPS
            elapsed = curr_time - start_time
            if elapsed > 0:
                fps = frame_count / elapsed

            # Log data ke CSV
            if csv_writer:
                csv_writer.writerow([
                    datetime.now().isoformat(),
                    frame_count,
                    f"{fps:.2f}",
                    f"{latency_ms:.2f}",
                    landmark_valid,
                    f"{ear_val:.4f}",
                    f"{mar_val:.4f}",
                    f"{pitch_ratio:.4f}",
                    f"{yaw_ratio:.4f}",
                    f"{raw_drowsy_prob:.4f}" if raw_drowsy_prob is not None else "",
                    f"{smoothed_drowsy_prob:.4f}" if smoothed_drowsy_prob is not None else "",
                    status,
                    scenario_label,
                    f"{fatigue_duration:.2f}",
                    alarm_active
                ])

            # Rendering UI Visual & Dashboard
            if not no_display:
                draw_visuals(
                    frame=frame,
                    landmarks=landmarks,
                    bbox=bbox,
                    status_color=status_color,
                    is_nodding=is_nodding,
                    is_looking_aside=is_looking_aside,
                    head_direction=head_direction
                )
                draw_dashboard_overlay(
                    frame=frame,
                    fps=fps,
                    latency_ms=latency_ms,
                    ear=ear_val,
                    mar=mar_val,
                    pitch_ratio=pitch_ratio,
                    yaw_ratio=yaw_ratio,
                    head_direction=head_direction,
                    raw_drowsy_prob=raw_drowsy_prob,
                    smoothed_drowsy_prob=smoothed_drowsy_prob,
                    status=status,
                    scenario_label=scenario_label,
                    status_color=status_color,
                    landmark_valid=landmark_valid,
                    buffer_len=len(buffer),
                    buffer_max=seq_len,
                    fatigue_duration=fatigue_duration,
                    target_alert_sec=current_target_alert_sec,
                    alarm_active=alarm_active
                )

                cv2.imshow("Real-Time Drowsiness Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[*] 'q' key pressed. Exiting...")
                    break

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user. Closing...")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        if csv_file_obj:
            csv_file_obj.close()

        total_time = time.time() - start_time
        print("\n" + "=" * 65)
        print("                      SESSION SUMMARY                    ")
        print("=" * 65)
        print(f"Total Running Time      : {total_time:.2f} s")
        print(f"Total Frames Processed  : {frame_count}")
        print(f"Valid Landmark Frames   : {valid_frames} ({valid_frames/max(1, frame_count)*100:.1f}%)")
        print(f"Invalid Landmark Frames : {invalid_frames} ({invalid_frames/max(1, frame_count)*100:.1f}%)")
        if total_time > 0:
            print(f"Average FPS             : {frame_count/total_time:.2f}")
        print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Drowsiness Detection via Webcam")
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX, help="Index kamera OpenCV (default: 0)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Threshold klasifikasi DROWSY (default: 0.50)")
    parser.add_argument("--smoothing-window", type=int, default=SMOOTHING_WINDOW, help="Ukuran window moving average (default: 5)")
    parser.add_argument("--drowsy-alert-sec", type=float, default=DROWSY_ALERT_SECONDS, help="Batas durasi kantuk umum (detik) sebelum alarm suara berbunyi (default: 4.0)")
    parser.add_argument("--nodding-alert-sec", type=float, default=NODDING_ALERT_SECONDS, help="Batas durasi kepala nodding/menunduk (detik) sebelum alarm suara aktif (default: 1.5)")
    parser.add_argument("--look-aside-alert-sec", type=float, default=LOOK_ASIDE_ALERT_SECONDS, help="Batas durasi menengok kiri/kanan (detik) sebelum alarm aktif (default: 3.0)")
    parser.add_argument("--microsleep-alert-sec", type=float, default=MICROSLEEP_ALERT_SECONDS, help="Batas durasi mata terpejam / microsleep (detik) sebelum alarm suara aktif (default: 1.0)")
    parser.add_argument("--yawn-alert-sec", type=float, default=YAWN_ALERT_SECONDS, help="Batas durasi menguap (detik) sebelum alarm suara aktif (default: 4.5)")
    parser.add_argument("--eye-thresh", type=float, default=EYE_CLOSED_EAR_THRESHOLD, help="Batas EAR mata terpejam (default: 0.21)")
    parser.add_argument("--yawn-thresh", type=float, default=YAWN_MAR_THRESHOLD, help="Batas MAR menguap (default: 0.45)")
    parser.add_argument("--nodding-thresh", type=float, default=NODDING_PITCH_RATIO_THRESHOLD, help="Batas rasio vertikal wajah untuk deteksi nodding (default: 0.45)")
    parser.add_argument("--look-aside-thresh", type=float, default=LOOK_ASIDE_YAW_THRESHOLD, help="Batas rasio rotasi horizontal untuk menengok kiri/kanan (default: 0.28)")
    parser.add_argument("--no-sound", action="store_true", help="Nonaktifkan bunyi suara alarm kantuk")
    parser.add_argument("--device", type=str, default="cpu", help="Device PyTorch (cpu/cuda, default: cpu)")
    parser.add_argument("--log", action="store_true", help="Simpan riwayat prediksi ke logs/realtime_predictions.csv")
    parser.add_argument("--log-file", type=str, default=DEFAULT_LOG_PATH, help="Custom path file log CSV")
    parser.add_argument("--no-display", action="store_true", help="Jalankan tanpa jendela GUI OpenCV (headless mode)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_dest = args.log_file if args.log else None
    enable_audio = not args.no_sound
    run_realtime(
        camera_index=args.camera_index,
        threshold=args.threshold,
        smoothing_window=args.smoothing_window,
        drowsy_alert_sec=args.drowsy_alert_sec,
        nodding_alert_sec=args.nodding_alert_sec,
        look_aside_alert_sec=args.look_aside_alert_sec,
        microsleep_alert_sec=args.microsleep_alert_sec,
        yawn_alert_sec=args.yawn_alert_sec,
        eye_thresh=args.eye_thresh,
        yawn_thresh=args.yawn_thresh,
        nodding_ratio_thresh=args.nodding_thresh,
        look_aside_yaw_thresh=args.look_aside_thresh,
        enable_sound=enable_audio,
        device=args.device,
        log_file=log_dest,
        no_display=args.no_display
    )
