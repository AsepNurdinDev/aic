"""
src/feature_extractor.py

Ekstraksi fitur EAR (Eye Aspect Ratio) dan MAR (Mouth Aspect Ratio) dari facial landmarks.
WAJIB menggunakan implementasi EAR V3 yang SAMA PERSIS dengan model training.

Landmark Indices:
- LEFT_EYE  = [33, 133, 159, 145]  (p1: outer, p2: inner, p3: upper, p4: lower)
- RIGHT_EYE = [362, 263, 386, 374] (p1: outer, p2: inner, p3: upper, p4: lower)
- MOUTH     = [61, 291, 13, 14]     (left: 61, right: 291, upper: 13, lower: 14)

Diagnostic EAR V3:
- Rentang normal: ~0.10 – 0.44
- EAR > 1.0 -> INVALID
- MAR > 1.0 -> INVALID
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
import numpy as np

# Landmark indices persis dari notebook training
LEFT_EYE: List[int] = [33, 133, 159, 145]
RIGHT_EYE: List[int] = [362, 263, 386, 374]
MOUTH: List[int] = [61, 291, 13, 14]


def _get_coords(landmark: Any) -> Tuple[float, float]:
    """Helper untuk mengambil koordinat (x, y) dari landmark object atau dict/tuple."""
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        return float(landmark.x), float(landmark.y)
    elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
        return float(landmark[0]), float(landmark[1])
    elif isinstance(landmark, dict) and "x" in landmark and "y" in landmark:
        return float(landmark["x"]), float(landmark["y"])
    raise ValueError(f"Unsupported landmark format: {type(landmark)}")


def distance_2d(a: Any, b: Any) -> float:
    """
    Hitung jarak Euclidean 2D antara dua landmark.
    Formula: sqrt((a.x - b.x)^2 + (a.y - b.y)^2)
    """
    ax, ay = _get_coords(a)
    bx, by = _get_coords(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def calculate_ear(landmarks: Sequence, indices: List[int]) -> float:
    """
    Hitung EAR untuk satu mata menggunakan indeks landmark yang ditentukan.
    indices: [p1, p2, p3, p4]
      p1: sudut mata luar
      p2: sudut mata dalam
      p3: kelopak mata atas
      p4: kelopak mata bawah
    """
    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]

    horizontal = distance_2d(p1, p2)
    vertical = distance_2d(p3, p4)

    if horizontal < 1e-8:
        return 0.0

    return vertical / horizontal


def calculate_both_eyes(landmarks: Any) -> Tuple[float, float, float]:
    """
    Hitung EAR mata kiri, mata kanan, dan rata-rata keduanya.
    """
    left_ear = calculate_ear(landmarks, LEFT_EYE)
    right_ear = calculate_ear(landmarks, RIGHT_EYE)
    mean_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, mean_ear


def calculate_mar(landmarks: Any) -> float:
    """
    Hitung MAR (Mouth Aspect Ratio).
    Indeks:
      left corner  = 61
      right corner = 291
      upper lip    = 13
      lower lip    = 14
    """
    left = landmarks[MOUTH[0]]
    right = landmarks[MOUTH[1]]
    upper = landmarks[MOUTH[2]]
    lower = landmarks[MOUTH[3]]

    mouth_width = distance_2d(left, right)
    mouth_height = distance_2d(upper, lower)

    if mouth_width < 1e-8:
        return 0.0

    return mouth_height / mouth_width


def extract_features(landmarks: Optional[Any]) -> Dict[str, Union[float, bool]]:
    """
    Ekstrak fitur mentah EAR dan MAR dari landmarks wajah.

    Args:
        landmarks: List normalized landmarks dari MediaPipe.

    Returns:
        Dict berisi:
            - "ear": float (Mean EAR)
            - "mar": float (MAR)
            - "left_ear": float
            - "right_ear": float
            - "valid": bool
    """
    if landmarks is None or len(landmarks) == 0:
        return {
            "ear": 0.0,
            "mar": 0.0,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "valid": False
        }

    try:
        left_ear, right_ear, ear = calculate_both_eyes(landmarks)
        mar = calculate_mar(landmarks)

        # Validasi:
        # 1. Pastikan finite (bukan NaN atau Inf)
        if not (math.isfinite(ear) and math.isfinite(mar) and math.isfinite(left_ear) and math.isfinite(right_ear)):
            return {
                "ear": 0.0,
                "mar": 0.0,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "valid": False
            }

        # 2. Tidak boleh bernilai negatif
        if ear < 0.0 or mar < 0.0 or left_ear < 0.0 or right_ear < 0.0:
            return {
                "ear": float(ear),
                "mar": float(mar),
                "left_ear": float(left_ear),
                "right_ear": float(right_ear),
                "valid": False
            }

        # 3. EAR > 1.0 atau MAR > 1.0 dianggap tidak valid (anomali deteksi)
        if ear > 1.0 or mar > 1.0:
            return {
                "ear": float(ear),
                "mar": float(mar),
                "left_ear": float(left_ear),
                "right_ear": float(right_ear),
                "valid": False
            }

        return {
            "ear": float(ear),
            "mar": float(mar),
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "valid": True
        }

    except Exception:
        return {
            "ear": 0.0,
            "mar": 0.0,
            "left_ear": 0.0,
            "right_ear": 0.0,
            "valid": False
        }
