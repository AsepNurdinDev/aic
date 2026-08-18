"""
backend/src/feature_extractor.py

Ekstraksi fitur EAR (Eye Aspect Ratio V3) dan MAR (Mouth Aspect Ratio) dari facial landmarks.
Landmark Indices:
- LEFT_EYE  = [33, 133, 159, 145]  (p1: outer, p2: inner, p3: upper, p4: lower)
- RIGHT_EYE = [362, 263, 386, 374] (p1: outer, p2: inner, p3: upper, p4: lower)
- MOUTH     = [61, 291, 13, 14]     (left: 61, right: 291, upper: 13, lower: 14)
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

LEFT_EYE: List[int] = [33, 133, 159, 145]
RIGHT_EYE: List[int] = [362, 263, 386, 374]
MOUTH: List[int] = [61, 291, 13, 14]

# Head pose landmarks
LM_FOREHEAD = 10
LM_NOSE = 1
LM_CHIN = 152
LM_LEFT_CHEEK = 234
LM_RIGHT_CHEEK = 454
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263


def _get_coords(landmark: Any) -> Tuple[float, float]:
    if hasattr(landmark, "x") and hasattr(landmark, "y"):
        return float(landmark.x), float(landmark.y)
    elif isinstance(landmark, (list, tuple)) and len(landmark) >= 2:
        return float(landmark[0]), float(landmark[1])
    elif isinstance(landmark, dict) and "x" in landmark and "y" in landmark:
        return float(landmark["x"]), float(landmark["y"])
    raise ValueError(f"Unsupported landmark format: {type(landmark)}")


def distance_2d(a: Any, b: Any) -> float:
    ax, ay = _get_coords(a)
    bx, by = _get_coords(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def calculate_ear(landmarks: Sequence, indices: List[int]) -> float:
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
    left_ear = calculate_ear(landmarks, LEFT_EYE)
    right_ear = calculate_ear(landmarks, RIGHT_EYE)
    mean_ear = (left_ear + right_ear) / 2.0
    return left_ear, right_ear, mean_ear


def calculate_mar(landmarks: Any) -> float:
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

        if not (math.isfinite(ear) and math.isfinite(mar) and math.isfinite(left_ear) and math.isfinite(right_ear)):
            return {
                "ear": 0.0,
                "mar": 0.0,
                "left_ear": 0.0,
                "right_ear": 0.0,
                "valid": False
            }

        if ear < 0.0 or mar < 0.0 or left_ear < 0.0 or right_ear < 0.0:
            return {
                "ear": float(ear),
                "mar": float(mar),
                "left_ear": float(left_ear),
                "right_ear": float(right_ear),
                "valid": False
            }

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


def evaluate_head_pose(
    landmarks: Any,
    nodding_ratio_threshold: float = 0.45,
    yaw_ratio_threshold: float = 0.28,
    ear_val: float = 0.30,
    smoothed_drowsy_prob: float = 0.0
) -> Tuple[bool, bool, float, float, str]:
    """
    Evaluasi orientasi kepala (Nodding & Looking Aside).
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

        # 1. Pitch Vertikal (Dahi-Hidung-Dagu)
        upper_dist = abs(p_nose.y - p_forehead.y)
        lower_dist = abs(p_chin.y - p_nose.y)
        pitch_ratio = lower_dist / max(1e-6, upper_dist)

        # 2. Yaw Horizontal 2D
        dist_l = abs(p_nose.x - p_left_cheek.x)
        dist_r = abs(p_right_cheek.x - p_nose.x)
        total_w = dist_l + dist_r
        yaw_ratio = (dist_l - dist_r) / max(1e-6, total_w)

        # 3. Sudut Yaw 3D
        yaw_deg = math.degrees(math.atan2(p_left_eye.z - p_right_eye.z, max(1e-6, abs(p_right_eye.x - p_left_eye.x))))

        is_deep_nod = (pitch_ratio < nodding_ratio_threshold)
        is_drowsy_nod = (pitch_ratio < (nodding_ratio_threshold + 0.07)) and (ear_val < 0.25 or smoothed_drowsy_prob > 0.40)
        is_nodding = bool(is_deep_nod or is_drowsy_nod)

        is_looking_aside = bool((abs(yaw_ratio) >= yaw_ratio_threshold) or (abs(yaw_deg) >= 18.0))

        direction = "CENTER"
        if is_looking_aside:
            direction = "LEFT" if (yaw_ratio < 0 or yaw_deg > 0) else "RIGHT"
        elif is_nodding:
            direction = "DOWN"

        return is_nodding, is_looking_aside, float(pitch_ratio), float(yaw_ratio), direction

    except Exception:
        return False, False, 1.0, 0.0, "CENTER"
