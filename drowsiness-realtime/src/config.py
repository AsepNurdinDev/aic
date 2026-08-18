"""
src/config.py

Pusat konfigurasi aplikasi real-time drowsiness detection.
Menggunakan parameter model dan checkpoint yang telah dilatih sebagai source of truth.
"""

import os
import json
from typing import Dict, List, Tuple
import numpy as np

# Path direktori
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path model & assets
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_landmark_gru_ear_mar.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "FINAL_landmark_gru_config.json")
MEDIAPIPE_MODEL_PATH = os.path.join(BASE_DIR, "mediapipe", "face_landmarker.task")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DEFAULT_LOG_PATH = os.path.join(LOGS_DIR, "realtime_predictions.csv")

# Parameter inferensi default
SEQUENCE_LENGTH = 60
THRESHOLD = 0.50
SMOOTHING_WINDOW = 5
CAMERA_INDEX = 0

# Parameter Audio Alert & Skenario Kantuk Realistis
DROWSY_ALERT_SECONDS = 4.0        # Durasi kantuk umum (detik) sebelum suara alarm aktif (4 detik)
NODDING_ALERT_SECONDS = 1.5       # Durasi saat kepala nodding / menunduk terkulai (1.5 detik)
LOOK_ASIDE_ALERT_SECONDS = 3.0    # Durasi menengok kiri/kanan (distraction) sebelum alert (3 detik)
MICROSLEEP_ALERT_SECONDS = 1.0    # Respon cepat: Mata terpejam langsung memicu alarm dalam 1.0 detik
YAWN_ALERT_SECONDS = 4.5          # Respon toleran: Menguap tidak langsung alarm kecuali berlangsung > 4.5 detik

EYE_CLOSED_EAR_THRESHOLD = 0.21   # Batas EAR mata terpejam / microsleep
YAWN_MAR_THRESHOLD = 0.45         # Batas MAR mulut terbuka lebar / menguap
NODDING_PITCH_RATIO_THRESHOLD = 0.45 # Batas rasio vertikal wajah untuk nodding (dikalibrasi)
LOOK_ASIDE_YAW_THRESHOLD = 0.28   # Batas rasio rotasi horizontal (menengok kiri / kanan ~20 derajat)

ALARM_BEEP_FREQUENCY = 1500       # Frekuensi bunyi beep (Hz)
ALARM_BEEP_DURATION_MS = 350      # Durasi bunyi beep per pulsa (ms)
ENABLE_AUDIO_ALERT = True         # Status aktif/non-aktif suara alarm secara default

# Mapping kelas
CLASS_NAMES: Dict[int, str] = {
    0: "NOT DROWSY",
    1: "DROWSY"
}

# Warna visualisasi BGR
COLOR_NOT_DROWSY = (0, 200, 0)      # Hijau (Sadar / Waspada)
COLOR_DROWSY = (0, 0, 255)          # Merah (Kantuk Terdeteksi)
COLOR_NODDING = (0, 0, 255)         # Merah (Nodding Terdeteksi)
COLOR_LOOK_ASIDE = (0, 140, 255)    # Oranye-Merah (Menengok Kiri / Kanan)
COLOR_YAWNING = (0, 165, 255)       # Oranye (Menguap / Peringatan Dini)
COLOR_MICROSLEEP = (0, 0, 255)      # Merah (Mata Terpejam Kritis)
COLOR_BUFFERING = (255, 165, 0)     # Kuning-Oranye
COLOR_UNKNOWN = (128, 128, 128)     # Abu-abu
COLOR_TEXT = (255, 255, 255)        # Putih
COLOR_OVERLAY_BG = (20, 20, 20)     # Gelap transparan

# Training statistics (Mean & Std)
# Diambil langsung dari metadata checkpoint training
DEFAULT_FEATURE_MEAN: List[float] = [0.32720324397087097, 0.2745433449745178]
DEFAULT_FEATURE_STD: List[float] = [0.08445222675800323, 0.1241738498210907]


def get_feature_stats() -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengambil feature mean dan std dari checkpoint atau config file.
    """
    mean = np.array(DEFAULT_FEATURE_MEAN, dtype=np.float32)
    std = np.array(DEFAULT_FEATURE_STD, dtype=np.float32)

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "feature_mean" in cfg:
                    mean = np.array(cfg["feature_mean"], dtype=np.float32)
                if "feature_std" in cfg:
                    std = np.array(cfg["feature_std"], dtype=np.float32)
        except Exception:
            pass

    return mean, std
