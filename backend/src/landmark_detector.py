"""
backend/src/landmark_detector.py

Deteksi wajah dan 478 facial landmarks menggunakan MediaPipe Tasks Face Landmarker API modern.
"""

import os
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarkDetector:
    """
    Wrapper untuk MediaPipe Face Landmarker Tasks API.
    """

    def __init__(
        self,
        model_path: str = "mediapipe/face_landmarker.task",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at: {model_path}."
            )

        self.model_path = model_path
        self.num_faces = num_faces

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_face_landmarks(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Deteksi landmark wajah dari frame OpenCV BGR.

        Returns:
            Dict berisi:
                - "valid": bool (True jika 1 wajah terdeteksi valid)
                - "landmarks": List landmarks (NormalizedLandmark) atau None
                - "bbox": (x, y, w, h) bounding box relatif/pixel atau None
                - "error_message": str
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "valid": False,
                "landmarks": None,
                "bbox": None,
                "error_message": "Empty or None frame provided."
            }

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            detection_result = self.landmarker.detect(mp_image)

            if not detection_result or not detection_result.face_landmarks:
                return {
                    "valid": False,
                    "landmarks": None,
                    "bbox": None,
                    "error_message": "No face detected."
                }

            if len(detection_result.face_landmarks) == 0:
                return {
                    "valid": False,
                    "landmarks": None,
                    "bbox": None,
                    "error_message": "Empty landmarks detected."
                }

            landmarks = detection_result.face_landmarks[0]

            if len(landmarks) < 468:
                return {
                    "valid": False,
                    "landmarks": None,
                    "bbox": None,
                    "error_message": f"Incomplete landmarks: {len(landmarks)}"
                }

            h, w = frame_bgr.shape[:2]
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x_min, x_max = max(0, int(min(xs))), min(w, int(max(xs)))
            y_min, y_max = max(0, int(min(ys))), min(h, int(max(ys)))
            bbox = (x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min))

            return {
                "valid": True,
                "landmarks": landmarks,
                "bbox": bbox,
                "error_message": ""
            }

        except Exception as e:
            return {
                "valid": False,
                "landmarks": None,
                "bbox": None,
                "error_message": f"Error during landmark detection: {str(e)}"
            }

    def close(self):
        """Release MediaPipe detector resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()
