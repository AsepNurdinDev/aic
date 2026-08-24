import { useState, useRef, useEffect, useCallback } from "react";
import UploadPage from "./UploadPage";
import { analyzeFrame, resetSession, checkHealth } from "../services/api";
import { audioAlert } from "../services/audioAlert";
import "../App.css";

function VideoCard({ label, source, videoRef, canvasRef, onPause, onEnded, isDriver, hasSource }) {
  useEffect(() => {
    if (source?.type === "camera" && videoRef.current) {
      videoRef.current.srcObject = source.stream;
    }
  }, [source, videoRef]);

  return (
    <div className="video-card">
      {source?.type === "file" && (
        <>
          <video
            ref={videoRef}
            src={source.url}
            className="video-player"
            playsInline
            onPause={onPause}
            onEnded={onEnded}
          />
          {isDriver && <canvas ref={canvasRef} className="video-overlay-canvas" />}
        </>
      )}

      {source?.type === "camera" && (
        <>
          <video
            ref={videoRef}
            className="video-player"
            autoPlay
            muted
            playsInline
          />
          {isDriver && <canvas ref={canvasRef} className="video-overlay-canvas" />}
          <span className="live-badge">LIVE CAMERA</span>
        </>
      )}

      {!source && <h2>{label}</h2>}
    </div>
  );
}

function HomePage() {
  const [showUpload, setShowUpload] = useState(false);
  const [driverSource, setDriverSource] = useState(null); // {type:'file',url} | {type:'camera',stream} | null
  const [roadSource, setRoadSource] = useState(null); // {type:'file',url} | null
  const [isPlaying, setIsPlaying] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");

  // State hasil analisis real-time
  const [analysis, setAnalysis] = useState(null);

  const driverRef = useRef(null);
  const roadRef = useRef(null);
  const driverCanvasRef = useRef(null);
  const offscreenCanvasRef = useRef(null);
  const isAnalyzingRef = useRef(false);
  const animationFrameRef = useRef(null);

  // Inisialisasi offscreen canvas untuk capture frame
  useEffect(() => {
    offscreenCanvasRef.current = document.createElement("canvas");
    let attempts = 0;
    const maxAttempts = 10;
    const tryHealth = async () => {
      const res = await checkHealth();
      if (res.model_ready) {
        setBackendStatus("ready");
      } else {
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(tryHealth, 3000);
        } else {
          setBackendStatus("error");
        }
      }
    };
    tryHealth();
  }, []);

  const handleUpload = ({ driver, road }) => {
    resetSession();
    audioAlert.stopAlarm();
    setAnalysis(null);

    if (driver?.type === "file") {
      setDriverSource({ type: "file", url: URL.createObjectURL(driver.file) });
    } else if (driver?.type === "camera") {
      setDriverSource({ type: "camera", stream: driver.stream });
    } else {
      setDriverSource(null);
    }

    if (road?.type === "file") {
      setRoadSource({ type: "file", url: URL.createObjectURL(road.file) });
    } else {
      setRoadSource(null);
    }

    setShowUpload(false);
    setIsPlaying(false);
  };

  // Matikan kamera kalau HomePage-nya sendiri unmount
  useEffect(() => {
    return () => {
      if (driverSource?.type === "camera" && driverSource.stream) {
        driverSource.stream.getTracks().forEach((t) => t.stop());
      }
      audioAlert.stopAlarm();
    };
  }, [driverSource]);

  // Tombol play/pause untuk video file
  const handleTogglePlay = () => {
    const playableRefs = [];
    if (driverSource?.type === "file" && driverRef.current) playableRefs.push(driverRef.current);
    if (roadSource?.type === "file" && roadRef.current) playableRefs.push(roadRef.current);

    if (playableRefs.length === 0) return;

    if (isPlaying) {
      playableRefs.forEach((v) => v.pause());
      setIsPlaying(false);
      audioAlert.stopAlarm();
    } else {
      playableRefs.forEach((v) => {
        v.play();
      });
      setIsPlaying(true);
    }
  };

  const handleAnyPause = () => {
    setIsPlaying(false);
    audioAlert.stopAlarm();
  };

  // Render visual landmarks dan bounding box pada overlay canvas
  const drawOverlay = useCallback((data) => {
    const canvas = driverCanvasRef.current;
    const video = driverRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = video.clientWidth || video.videoWidth || 640;
    const height = video.clientHeight || video.videoHeight || 360;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.clearRect(0, 0, width, height);

    if (!data || !data.face_detected) return;

    // Tentukan warna status
    let strokeColor = "#22c1d6";
    if (data.scenario === "MICROSLEEP" || data.scenario === "NODDING" || data.scenario === "DROWSY") {
      strokeColor = "#e74c3c"; // Merah
    } else if (data.scenario === "LOOK_ASIDE" || data.scenario === "YAWNING") {
      strokeColor = "#f39c12"; // Oranye
    } else if (data.scenario === "NORMAL") {
      strokeColor = "#2ecc71"; // Hijau
    }

    // 1. Gambar Bounding Box Wajah
    if (data.bbox) {
      const bx = data.bbox.x * width;
      const by = data.bbox.y * height;
      const bw = data.bbox.w * width;
      const bh = data.bbox.h * height;

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(bx, by, bw, bh);

      // Label skenario di atas bounding box
      ctx.fillStyle = strokeColor;
      ctx.font = "bold 13px 'Segoe UI', Arial, sans-serif";
      const tagText = data.is_nodding
        ? "NODDING DETECTED"
        : data.is_looking_aside
        ? `LOOKING ${data.head_direction}`
        : data.scenario_label || "";
      if (tagText) {
        ctx.fillText(tagText, bx, Math.max(18, by - 6));
      }
    }

    // 2. Gambar Titik Landmark Mata (Kuning-Cyan)
    if (data.key_landmarks?.left_eye) {
      ctx.fillStyle = "#f1c40f";
      data.key_landmarks.left_eye.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x * width, pt.y * height, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
    if (data.key_landmarks?.right_eye) {
      ctx.fillStyle = "#f1c40f";
      data.key_landmarks.right_eye.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x * width, pt.y * height, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // 3. Gambar Titik Landmark Mulut (Magenta)
    if (data.key_landmarks?.mouth) {
      ctx.fillStyle = "#e056fd";
      data.key_landmarks.mouth.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x * width, pt.y * height, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }

    // 4. Gambar Titik Reference Head Pose (Oranye)
    if (data.key_landmarks?.face_axes) {
      ctx.fillStyle = "#ff793f";
      data.key_landmarks.face_axes.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x * width, pt.y * height, 3, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }, []);

  // Frame processing loop untuk Driver (Webcam / Video Playback)
  useEffect(() => {
    let intervalId = null;
    const shouldRun =
      driverSource?.type === "camera" ||
      (driverSource?.type === "file" && isPlaying);

    if (shouldRun) {
      intervalId = setInterval(async () => {
        if (isAnalyzingRef.current) return;
        const video = driverRef.current;
        const offCanvas = offscreenCanvasRef.current;

        if (!video || video.readyState < 2 || video.videoWidth === 0) return;

        isAnalyzingRef.current = true;
        try {
          // Resize offscreen canvas untuk efisiensi transfer data (~320x240)
          const targetW = 320;
          const targetH = Math.round((video.videoHeight / video.videoWidth) * targetW) || 240;
          offCanvas.width = targetW;
          offCanvas.height = targetH;

          const ctx = offCanvas.getContext("2d");
          ctx.drawImage(video, 0, 0, targetW, targetH);
          const dataUrl = offCanvas.toDataURL("image/jpeg", 0.65);

          const result = await analyzeFrame(dataUrl);

          setAnalysis(result);
          drawOverlay(result);

          // Pemicu Alarm Audio
          if (result.alarm_active) {
            audioAlert.startAlarm();
          } else {
            audioAlert.stopAlarm();
          }
        } catch (e) {
          console.warn("Frame analysis error:", e);
        } finally {
          isAnalyzingRef.current = false;
        }
      }, 90); // ~11 FPS (responsif & hemat resource)
    } else {
      audioAlert.stopAlarm();
      // Bersihkan canvas jika video pause atau tidak aktif
      if (driverCanvasRef.current) {
        const ctx = driverCanvasRef.current.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, driverCanvasRef.current.width, driverCanvasRef.current.height);
      }
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
      audioAlert.stopAlarm();
    };
  }, [driverSource, isPlaying, drawOverlay]);

  const hasDriver = !!driverSource;
  const hasPlayableFile =
    driverSource?.type === "file" || roadSource?.type === "file";

  // Helper untuk warna Risk Level & Status
  const getRiskClass = (level) => {
    switch (level) {
      case "CRITICAL":
        return "risk-critical";
      case "WARNING":
        return "risk-warning";
      case "POTENTIAL":
        return "risk-potential";
      default:
        return "risk-low";
    }
  };

  const getStatusColor = (scenario) => {
    switch (scenario) {
      case "MICROSLEEP":
      case "NODDING":
      case "DROWSY":
        return "#e74c3c";
      case "LOOK_ASIDE":
      case "YAWNING":
        return "#f39c12";
      case "NORMAL":
        return "#2ecc71";
      case "BUFFERING":
        return "#22c1d6";
      default:
        return "#7f8c8d";
    }
  };

  return (
    <main className="home">
      <header className="header">
        <div className="header-title-wrapper">
          <h1>AIC — SafeRoute AI</h1>
          <span className={`backend-indicator ${backendStatus === "ready" ? "ready" : "offline"}`}>
            {backendStatus === "ready" ? "AI Engine Ready" : "AI Offline"}
          </span>
        </div>
        <button onClick={() => setShowUpload(true)}>UPLOAD</button>
      </header>

      <section className="video-section">
        <VideoCard
          label="driver video"
          source={driverSource}
          videoRef={driverRef}
          canvasRef={driverCanvasRef}
          onPause={handleAnyPause}
          onEnded={handleAnyPause}
          isDriver={true}
          hasSource={!!driverSource}
        />

        {hasDriver && hasPlayableFile && (
          <button
            className="play-both-button"
            onClick={handleTogglePlay}
            aria-label={isPlaying ? "Pause videos" : "Play videos"}
          >
            {isPlaying ? (
              <svg viewBox="0 0 24 24" fill="currentColor">
                <rect x="6" y="5" width="4" height="14" />
                <rect x="14" y="5" width="4" height="14" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>
        )}

        <VideoCard
          label="road video"
          source={roadSource}
          videoRef={roadRef}
          onPause={handleAnyPause}
          onEnded={handleAnyPause}
          isDriver={false}
          hasSource={!!roadSource}
        />
      </section>

      {/* Bagian Current Condition Terintegrasi */}
      <section className="condition">
        <div className="condition-header">
          <div className="condition-title-group">
            <h2>Current Condition</h2>
            {analysis ? (
              <span
                className="condition-scenario-pill"
                style={{ backgroundColor: getStatusColor(analysis.scenario) }}
              >
                <span className="pulsing-dot"></span>
                {analysis.status || "MONITORING"}
              </span>
            ) : (
              <span className="condition-scenario-pill standby">
                {driverSource ? "STANDBY — READY" : "STANDBY — NO VIDEO"}
              </span>
            )}
          </div>

          <div className="risk-level-container">
            <span className="risk-label-text">Risk Level:</span>
            <span className={`risk-badge ${getRiskClass(analysis?.risk_level || "LOW")}`}>
              {analysis?.risk_level || "LOW"}
            </span>
            {analysis && (
              <span className="risk-score-pill">Score: {analysis.risk_score}/100</span>
            )}
          </div>
        </div>

        {/* Alarm Banner Visual */}
        {analysis?.alarm_active && (
          <div className="alarm-banner">
            <span className="alarm-icon">⚠️</span>
            <span className="alarm-text">
              ALARM ACTIVE: {analysis.scenario_label} MELEBIHI {analysis.target_alert_sec}s!
            </span>
          </div>
        )}

        {/* Real-time Metrics Grid */}
        <div className="metrics-grid">
          <div className="metric-box">
            <span className="metric-label">EAR (Eye Aspect Ratio)</span>
            <span className="metric-value">
              {analysis?.ear !== undefined ? analysis.ear.toFixed(3) : "--"}
            </span>
            <span className="metric-sub">
              {analysis?.ear < 0.21 ? "Closed (<0.21)" : "Normal Open"}
            </span>
          </div>

          <div className="metric-box">
            <span className="metric-label">MAR (Mouth Ratio)</span>
            <span className="metric-value">
              {analysis?.mar !== undefined ? analysis.mar.toFixed(3) : "--"}
            </span>
            <span className="metric-sub">
              {analysis?.mar >= 0.45 ? "Yawning (>=0.45)" : "Normal"}
            </span>
          </div>

          <div className="metric-box">
            <span className="metric-label">Head Pose & Direction</span>
            <span className="metric-value head-dir">
              {analysis?.head_direction || "CENTER"}
            </span>
            <span className="metric-sub">
              P: {analysis?.pitch_ratio !== undefined ? analysis.pitch_ratio.toFixed(2) : "1.00"} | Y: {analysis?.yaw_ratio !== undefined ? (analysis.yaw_ratio > 0 ? `+${analysis.yaw_ratio.toFixed(2)}` : analysis.yaw_ratio.toFixed(2)) : "0.00"}
            </span>
          </div>

          <div className="metric-box">
            <span className="metric-label">GRU Drowsy Confidence</span>
            <span className="metric-value">
              {analysis?.smoothed_drowsy_prob !== null && analysis?.smoothed_drowsy_prob !== undefined
                ? `${Math.round(analysis.smoothed_drowsy_prob * 100)}%`
                : analysis?.buffer_len
                ? `${analysis.buffer_len}/${analysis.buffer_max}`
                : "--"}
            </span>
            <span className="metric-sub">
              {analysis?.smoothed_drowsy_prob >= 0.5 ? "Drowsy Pattern" : "Alert / Normal"}
            </span>
          </div>

          <div className="metric-box">
            <span className="metric-label">Alert Duration Timer</span>
            <span className={`metric-value ${analysis?.alarm_active ? "text-danger" : ""}`}>
              {analysis?.fatigue_duration !== undefined
                ? `${analysis.fatigue_duration.toFixed(1)}s / ${analysis.target_alert_sec.toFixed(1)}s`
                : "0.0s / 4.0s"}
            </span>
            <span className="metric-sub">
              {analysis?.alarm_active ? "Threshold Exceeded" : "Safe Window"}
            </span>
          </div>
        </div>

        {/* AI Explainability / Risk Reasons */}
        <div className="condition-reasons-section">
          <p className="reasons-title">AI Safety Assessment & Reasons:</p>
          <ul className="reasons-list">
            {analysis?.risk_reasons && analysis.risk_reasons.length > 0 ? (
              analysis.risk_reasons.map((r, i) => <li key={i}>{r}</li>)
            ) : (
              <li>Kondisi pengemudi dalam status standby atau terpantau sadar dan aman.</li>
            )}
          </ul>
        </div>
      </section>

      {showUpload && (
        <UploadPage
          onClose={() => setShowUpload(false)}
          onUpload={handleUpload}
        />
      )}
    </main>
  );
}

export default HomePage;