"""
SafeRoute AI — FastAPI Backend Server
Synchronous REST API for Driver Safety AI inference.
No WebSocket, no background tasks, no job queue.
"""

import base64
import time
import cv2
import numpy as np
import tempfile
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.realtime.pipeline import RealtimePipeline
from src.realtime.recorder import SessionRecorder


# ─── Global State ──────────────────────────────────────────────
pipeline: RealtimePipeline | None = None
recorder: SessionRecorder = SessionRecorder()


# ─── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("\n[SERVER] Loading AI models...")
    try:
        pipeline = RealtimePipeline()
        print("[SERVER] Pipeline ready.\n")
    except Exception as e:
        print(f"[SERVER] Pipeline failed to load: {e}")
        pipeline = None
    yield
    print("[SERVER] Shutdown complete.")


# ─── App ───────────────────────────────────────────────────────
app = FastAPI(title="SafeRoute AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ───────────────────────────────────────────────────
def decode_base64_frame(data: str | None) -> np.ndarray | None:
    """Decode a base64-encoded image to BGR numpy array."""
    if not data:
        return None
    try:
        # Strip data URI prefix if present
        if "," in data:
            data = data.split(",", 1)[1]
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def build_response(decision: dict) -> dict:
    """Build the API response from a decision dict."""
    return {
        "warning_message": decision.get("warning_message"),
        "decision": {
            "mode": decision.get("decision_mode", "DEGRADED"),
            "severity": decision.get("severity", "DEGRADED"),
            "action": decision.get("action", "DEGRADED_WARNING"),
        }
    }


# ─── Request Models ───────────────────────────────────────────
class InferRequest(BaseModel):
    driver_frame: Optional[str] = None
    road_frame: Optional[str] = None
    cabin_frame: Optional[str] = None


# ════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """System health and model availability."""
    if pipeline is None:
        return JSONResponse({
            "status": "OFFLINE",
            "models": {
                "drowsiness": "ERROR",
                "statefarm": "ERROR",
                "road_object": "ERROR",
                "road_geometry": "ERROR",
                "decision": "ERROR"
            }
        })
    return JSONResponse({
        "status": "ONLINE",
        "models": {
            "drowsiness": "READY" if pipeline.drowsiness_processor else "ERROR",
            "statefarm": "READY" if pipeline.statefarm_processor else "ERROR",
            "road_object": "READY" if pipeline.road_object_processor else "ERROR",
            "road_geometry": "READY" if pipeline.road_geometry_processor else "ERROR",
            "decision": "READY" if pipeline.decision_engine else "ERROR"
        }
    })


@app.post("/api/infer")
async def infer_frames(req: InferRequest):
    """Synchronous inference on up to 3 base64-encoded frames.
    
    Each frame corresponds to a camera channel:
    - driver_frame: for drowsiness + statefarm
    - road_frame: for road_object + road_geometry
    - cabin_frame: reserved
    
    Returns warning_message and decision.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="AI pipeline not loaded")
    
    driver_bgr = decode_base64_frame(req.driver_frame)
    road_bgr = decode_base64_frame(req.road_frame)
    cabin_bgr = decode_base64_frame(req.cabin_frame)
    
    if driver_bgr is None and road_bgr is None and cabin_bgr is None:
        raise HTTPException(status_code=400, detail="No valid frames provided")
    
    decision = pipeline.process_channels(driver_bgr, road_bgr, cabin_bgr)
    
    # Record frame input and output data
    recorder.record_frame(
        inputs_meta={
            "driver_frame_received": driver_bgr is not None,
            "road_frame_received": road_bgr is not None,
            "cabin_frame_received": cabin_bgr is not None
        },
        decision=decision
    )
    
    return build_response(decision)


# ─── Recording & Data Export Endpoints ─────────────────────────

@app.post("/api/recording/start")
async def start_recording(session_name: Optional[str] = None):
    """Start a new recording session."""
    session_id = recorder.start_session(session_name=session_name)
    return {
        "status": "RECORDING_STARTED",
        "session_id": session_id,
        "start_time": recorder.start_time
    }


@app.post("/api/recording/stop")
async def stop_recording():
    """Stop current recording session and generate JSON dataset file."""
    result = recorder.stop_session()
    return result


@app.get("/api/recording/latest")
async def get_latest_recording():
    """Get or download latest recorded JSON data."""
    data = recorder.get_latest_data()
    if not data:
        raise HTTPException(status_code=404, detail="No recorded data found")
    return JSONResponse(data)


@app.get("/api/recording/download/{filename}")
async def download_recording(filename: str):
    """Download saved JSON recording dataset file."""
    filepath = recorder.output_dir / filename
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="Recording file not found")
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type="application/json"
    )


@app.get("/api/recording/list")
async def list_recordings():
    """List all saved session JSON files."""
    return {"recordings": recorder.list_saved_sessions()}


@app.post("/api/infer/video")
async def infer_video(
    driver_video: Optional[UploadFile] = File(None),
    road_video: Optional[UploadFile] = File(None),
    cabin_video: Optional[UploadFile] = File(None),
    sample_interval: int = Form(default=10)
):
    """Synchronous inference on uploaded video files.
    
    Processes videos frame-by-frame (sampled at `sample_interval` frames).
    Returns the final decision result after processing all frames.
    This is synchronous — the request blocks until processing is complete.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="AI pipeline not loaded")
    
    # Save uploaded files to temp
    temp_paths = {}
    try:
        for name, upload in [("driver", driver_video), ("road", road_video), ("cabin", cabin_video)]:
            if upload is not None:
                suffix = Path(upload.filename).suffix if upload.filename else ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                content = await upload.read()
                tmp.write(content)
                tmp.close()
                temp_paths[name] = tmp.name
        
        if not temp_paths:
            raise HTTPException(status_code=400, detail="No video files provided")
        
        # Open video captures
        caps = {}
        for name, path in temp_paths.items():
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                caps[name] = cap
        
        if not caps:
            raise HTTPException(status_code=400, detail="Could not open any video file")
        
        # Process frames synchronously
        frame_idx = 0
        last_decision = None
        results = []
        
        while True:
            frames = {}
            any_read = False
            
            for name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[name] = frame
                    any_read = True
            
            if not any_read:
                break
            
            # Sample every N frames
            if frame_idx % sample_interval == 0:
                decision = pipeline.process_channels(
                    driver_frame=frames.get("driver"),
                    road_frame=frames.get("road"),
                    cabin_frame=frames.get("cabin")
                )
                last_decision = decision
                
                # Collect any warnings
                if decision.get("warning_message"):
                    results.append({
                        "frame": frame_idx,
                        "warning_message": decision["warning_message"],
                        "decision": {
                            "mode": decision.get("decision_mode", "DEGRADED"),
                            "severity": decision.get("severity", "DEGRADED"),
                            "action": decision.get("action", "DEGRADED_WARNING"),
                        }
                    })
            
            frame_idx += 1
        
        # Close captures
        for cap in caps.values():
            cap.release()
        
        # Build final response
        if last_decision is None:
            return {"warning_message": None, "decision": {"mode": "DEGRADED", "severity": "DEGRADED", "action": "DEGRADED_WARNING"}, "events": [], "frames_processed": 0}
        
        return {
            "warning_message": last_decision.get("warning_message"),
            "decision": {
                "mode": last_decision.get("decision_mode", "DEGRADED"),
                "severity": last_decision.get("severity", "DEGRADED"),
                "action": last_decision.get("action", "DEGRADED_WARNING"),
            },
            "events": results,
            "frames_processed": frame_idx
        }
        
    finally:
        # Cleanup temp files
        for path in temp_paths.values():
            try:
                os.unlink(path)
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
