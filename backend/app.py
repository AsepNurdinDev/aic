"""
backend/app.py

AIC — SafeRoute AI Backend Server:
Mengintegrasikan deteksi kantuk real-time (V5 CNN-BiGRU-Attention) dengan API komunikasi React frontend.
Murni Synchronous, tanpa Logging permanen (sesuai aturan lomba).
"""

import os
import sys
import base64
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.driver_analyzer import DriverAnalyzer

app = Flask(__name__)
CORS(app)

# Path Model
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BACKEND_DIR, "models", "fl3d_v5_blinkaware_cnn_bigru_best.pth")
MEDIAPIPE_MODEL_PATH = os.path.join(BACKEND_DIR, "mediapipe", "face_landmarker.task")

print("[*] Initializing V5 AI Driver Analyzer...")
try:
    driver_analyzer = DriverAnalyzer(
        checkpoint_path=MODEL_PATH,
        mediapipe_model_path=MEDIAPIPE_MODEL_PATH,
        device="cpu"
    )
    print("[SUCCESS] V5 AI Driver Analyzer is ready.")
except Exception as e:
    print(f"[FATAL ERROR] Failed to initialize Driver Analyzer: {e}")
    driver_analyzer = None

@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "aic-backend",
        "model_ready": driver_analyzer is not None,
        "features": ["V5 CNN-BiGRU", "36-Features", "Throttled Inference", "Alarm Duration Gate"]
    })

@app.post("/api/session/reset")
def reset_session():
    if driver_analyzer:
        driver_analyzer.reset()
    return jsonify({"status": "ok", "message": "Session reset successfully."})

@app.post("/api/analyze/frame")
def analyze_frame():
    if driver_analyzer is None:
        return jsonify({"error": "Driver Analyzer is not initialized."}), 500

    frame_bgr = None

    if request.files:
        file_obj = request.files.get("file") or request.files.get("image") or request.files.get("frame")
        if file_obj:
            file_bytes = np.frombuffer(file_obj.read(), np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame_bgr is None and request.is_json:
        data = request.get_json(silent=True) or {}
        image_str = data.get("image") or data.get("frame")
        if image_str:
            if "," in image_str:
                image_str = image_str.split(",", 1)[1]
            try:
                img_data = base64.b64decode(image_str)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as err:
                return jsonify({"error": f"Failed to decode base64: {str(err)}"}), 400

    if frame_bgr is None:
        return jsonify({"error": "No valid image provided."}), 400

    try:
        analysis_result = driver_analyzer.process_frame(frame_bgr)
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({"error": f"Error during processing: {str(e)}"}), 500

if __name__ == "__main__":
    # Clean output only (No JSON/CSV logs written anywhere)
    app.run(host="0.0.0.0", port=5000, debug=False)