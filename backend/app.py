"""
backend/app.py

AIC — SafeRoute AI Backend Server:
Mengintegrasikan deteksi kantuk real-time (LandmarkGRU + MediaPipe Face Landmarker),
SafeRoute AI Risk Engine, dan API komunikasi dengan React frontend.
"""

import os
import sys
import base64
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Pastikan module backend dapat diakses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MODEL_PATH,
    CONFIG_PATH,
    MEDIAPIPE_MODEL_PATH,
    LOGS_DIR
)
from src.driver_analyzer import DriverAnalyzer

app = Flask(__name__)
CORS(app)

# Inisialisasi Driver Analyzer global
print("[*] Initializing SafeRoute AI Driver Analyzer...")
try:
    driver_analyzer = DriverAnalyzer(
        checkpoint_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        mediapipe_model_path=MEDIAPIPE_MODEL_PATH,
        device="cpu"
    )
    print("[SUCCESS] SafeRoute AI Driver Analyzer is ready.")
except Exception as e:
    print(f"[FATAL ERROR] Failed to initialize Driver Analyzer: {e}")
    driver_analyzer = None


@app.get("/api/health")
def health():
    """Health check & status modul AI."""
    model_ready = driver_analyzer is not None
    return jsonify({
        "status": "ok",
        "service": "aic-backend",
        "model_ready": model_ready,
        "features": ["LandmarkGRU", "MediaPipe FaceLandmarker", "RiskEngine", "EAR V3 + MAR", "HeadPose"]
    })


@app.post("/api/session/reset")
def reset_session():
    """Reset sequence buffer dan status pelacakan durasi kantuk."""
    if driver_analyzer:
        driver_analyzer.reset()
    return jsonify({
        "status": "ok",
        "message": "Driver analysis session reset successfully."
    })


@app.post("/api/analyze/frame")
def analyze_frame():
    """
    Analisis satu frame video/kamera pengemudi secara real-time.
    Menerima:
      1. JSON { "image": "data:image/jpeg;base64,..." } ATAU
      2. Form-data file (key: "file" atau "image" atau "frame")
    """
    if driver_analyzer is None:
        return jsonify({
            "error": "Driver Analyzer model is not initialized."
        }), 500

    frame_bgr = None

    # 1. Cek upload file multipart
    if request.files:
        file_obj = request.files.get("file") or request.files.get("image") or request.files.get("frame")
        if file_obj:
            file_bytes = np.frombuffer(file_obj.read(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. Cek JSON base64
    if frame_bgr is None and request.is_json:
        data = request.get_json(silent=True) or {}
        image_str = data.get("image") or data.get("frame")
        if image_str:
            # Hilangkan prefix data:image/...;base64, jika ada
            if "," in image_str:
                image_str = image_str.split(",", 1)[1]
            try:
                img_data = base64.b64decode(image_str)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as err:
                return jsonify({"error": f"Failed to decode base64 image: {str(err)}"}), 400

    if frame_bgr is None:
        return jsonify({
            "error": "No valid image provided. Send base64 'image' in JSON or multipart file."
        }), 400

    try:
        # Analisis frame melalui pipeline SafeRoute AI
        analysis_result = driver_analyzer.process_frame(frame_bgr)
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({
            "error": f"Error during frame processing: {str(e)}"
        }), 500


@app.post("/api/analyze/video")
def analyze_video():
    """
    Analisis video file pengemudi.
    Menerima file video (key: "driver_video"), melakukan sampling frame,
    dan mengembalikan ringkasan timeline risiko.
    """
    if driver_analyzer is None:
        return jsonify({"error": "Driver Analyzer model is not initialized."}), 500

    if "driver_video" not in request.files:
        return jsonify({"error": "Missing 'driver_video' file in request."}), 400

    video_file = request.files["driver_video"]
    temp_path = os.path.join(LOGS_DIR, f"temp_{int(time.time())}_{video_file.filename}")
    video_file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": "Could not open uploaded video file."}), 400

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps if fps > 0 else 0

    # Reset state analyzer untuk video baru
    driver_analyzer.reset()

    timeline = []
    max_risk_score = 0
    risk_events = []
    frame_idx = 0
    # Process sampling 1 frame per 0.2 detik (~5 FPS sampling)
    sample_step = max(1, int(fps / 5.0))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            timestamp = round(frame_idx / fps, 2)
            result = driver_analyzer.process_frame(frame)
            risk_score = result.get("risk_score", 0)
            if risk_score > max_risk_score:
                max_risk_score = risk_score

            timeline_entry = {
                "timestamp": timestamp,
                "frame": frame_idx,
                "status": result.get("status"),
                "scenario": result.get("scenario"),
                "risk_level": result.get("risk_level"),
                "risk_score": risk_score,
                "ear": result.get("ear"),
                "mar": result.get("mar"),
                "alarm_active": result.get("alarm_active")
            }
            timeline.append(timeline_entry)

            if result.get("alarm_active") or result.get("risk_level") in ["WARNING", "CRITICAL"]:
                risk_events.append({
                    "timestamp": timestamp,
                    "event": result.get("scenario_label"),
                    "risk_level": result.get("risk_level"),
                    "reasons": result.get("risk_reasons")
                })

        frame_idx += 1

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)

    overall_risk = "LOW"
    if max_risk_score >= 75:
        overall_risk = "CRITICAL"
    elif max_risk_score >= 45:
        overall_risk = "WARNING"
    elif max_risk_score >= 25:
        overall_risk = "POTENTIAL"

    return jsonify({
        "status": "completed",
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "fps": round(fps, 2),
        "overall_risk_level": overall_risk,
        "max_risk_score": max_risk_score,
        "risk_events_count": len(risk_events),
        "risk_events": risk_events[:20],
        "timeline_sample_count": len(timeline),
        "timeline": timeline
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )