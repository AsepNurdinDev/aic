import { useState, useRef, useCallback, useEffect } from 'react'
import { CameraPanel } from '../components/CameraPanel'
import { SessionControls } from '../components/SessionControls'
import { ApiService, InferResponse } from '../services/api'
import { voiceAlert } from '../services/voiceAlert'

type SessionMode = 'live' | 'demo'

/**
 * MonitorPage — the only page in the app.
 * 3 camera panels + session controls + voice-only output.
 * No dashboard, no analytics, no text display of AI results.
 */
export function MonitorPage() {
  // ─── State ──────────────────────────────────────────────
  const [mode, setMode] = useState<SessionMode>('demo')
  const [isRunning, setIsRunning] = useState(false)
  const [backendOnline, setBackendOnline] = useState(false)

  // Demo files
  const [driverFile, setDriverFile] = useState<File | null>(null)
  const [roadFile, setRoadFile] = useState<File | null>(null)
  const [cabinFile, setCabinFile] = useState<File | null>(null)

  // Video sources (object URLs for demo, MediaStreams for live)
  const [driverVideoSrc, setDriverVideoSrc] = useState<string>('')
  const [roadVideoSrc, setRoadVideoSrc] = useState<string>('')
  const [cabinVideoSrc, setCabinVideoSrc] = useState<string>('')

  // Live streams
  const [driverStream, setDriverStream] = useState<MediaStream | null>(null)

  // Voice indicator
  const [voiceMessage, setVoiceMessage] = useState<string | null>(null)
  const [voiceSeverity, setVoiceSeverity] = useState<string>('SAFE')

  // Refs for cleanup
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const driverVideoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const voiceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ─── Health Check ───────────────────────────────────────
  useEffect(() => {
    const checkHealth = async () => {
      const health = await ApiService.getHealth()
      setBackendOnline(health.status === 'ONLINE')
    }
    checkHealth()
    const hInterval = setInterval(checkHealth, 10000)
    return () => clearInterval(hInterval)
  }, [])

  // ─── Capture frame from video element ───────────────────
  const captureFrame = useCallback((videoEl: HTMLVideoElement | null): string | null => {
    if (!videoEl || videoEl.readyState < 2) return null

    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas')
    }
    const canvas = canvasRef.current
    canvas.width = videoEl.videoWidth || 640
    canvas.height = videoEl.videoHeight || 480

    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height)

    return canvas.toDataURL('image/jpeg', 0.7)
  }, [])

  // ─── Process inference response ─────────────────────────
  const handleInferResponse = useCallback((res: InferResponse) => {
    const played = voiceAlert.processResponse(res.warning_message, res.decision.action)
    
    if (played && res.warning_message) {
      setVoiceMessage(res.warning_message)
      setVoiceSeverity(res.decision.severity)
      
      // Clear voice indicator after 4 seconds
      if (voiceTimeoutRef.current) clearTimeout(voiceTimeoutRef.current)
      voiceTimeoutRef.current = setTimeout(() => {
        setVoiceMessage(null)
      }, 4000)
    }
  }, [])

  // ─── Start Live Session ─────────────────────────────────
  const startLive = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      })
      setDriverStream(stream)

      // Create a hidden video element for frame capture
      const video = document.createElement('video')
      video.srcObject = stream
      video.muted = true
      video.playsInline = true
      await video.play()
      driverVideoRef.current = video

      setIsRunning(true)

      // Send frames at ~3 FPS
      intervalRef.current = setInterval(async () => {
        const frame = captureFrame(driverVideoRef.current)
        if (!frame) return

        try {
          const res = await ApiService.inferFrames(frame, null, null)
          handleInferResponse(res)
        } catch {
          // Silently ignore inference errors
        }
      }, 333)
    } catch (err) {
      console.error('Camera access denied:', err)
    }
  }, [captureFrame, handleInferResponse])

  // ─── Start Demo Session ─────────────────────────────────
  const startDemo = useCallback(async () => {
    // Create object URLs for the video panels
    if (driverFile) setDriverVideoSrc(URL.createObjectURL(driverFile))
    if (roadFile) setRoadVideoSrc(URL.createObjectURL(roadFile))
    if (cabinFile) setCabinVideoSrc(URL.createObjectURL(cabinFile))

    setIsRunning(true)

    // Send video files to backend for batch inference
    try {
      const res = await ApiService.inferVideo(driverFile, roadFile, cabinFile)
      
      // Play the final warning if any
      if (res.warning_message) {
        handleInferResponse({
          warning_message: res.warning_message,
          decision: res.decision,
        })
      }

      // Also play any intermediate events
      if (res.events && Array.isArray(res.events)) {
        for (const evt of res.events) {
          if (evt.warning_message) {
            // Small delay between messages to allow TTS to finish
            await new Promise(resolve => setTimeout(resolve, 2000))
            handleInferResponse({
              warning_message: evt.warning_message,
              decision: evt.decision,
            })
          }
        }
      }
    } catch (err) {
      console.error('Demo inference error:', err)
    }
  }, [driverFile, roadFile, cabinFile, handleInferResponse])

  // ─── Stop Session ───────────────────────────────────────
  const stopSession = useCallback(() => {
    // Stop interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    // Stop camera stream
    if (driverStream) {
      driverStream.getTracks().forEach(t => t.stop())
      setDriverStream(null)
    }
    if (driverVideoRef.current) {
      driverVideoRef.current.pause()
      driverVideoRef.current.srcObject = null
      driverVideoRef.current = null
    }

    // Revoke object URLs
    if (driverVideoSrc) URL.revokeObjectURL(driverVideoSrc)
    if (roadVideoSrc) URL.revokeObjectURL(roadVideoSrc)
    if (cabinVideoSrc) URL.revokeObjectURL(cabinVideoSrc)
    setDriverVideoSrc('')
    setRoadVideoSrc('')
    setCabinVideoSrc('')

    // Stop voice
    voiceAlert.stop()
    setVoiceMessage(null)

    setIsRunning(false)
  }, [driverStream, driverVideoSrc, roadVideoSrc, cabinVideoSrc])

  // ─── Cleanup on unmount ─────────────────────────────────
  useEffect(() => {
    return () => {
      stopSession()
    }
  }, [])

  // ─── Start handler ──────────────────────────────────────
  const handleStart = () => {
    if (mode === 'live') startLive()
    else startDemo()
  }

  // ─── Render ─────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">SR</div>
          <h1 className="header-title">SafeRoute AI</h1>
        </div>
        <div className="header-status">
          <span className={`status-dot ${backendOnline ? 'online' : 'offline'}`} />
          <span style={{ color: backendOnline ? 'var(--accent-green)' : 'var(--accent-red)' }}>
            {backendOnline ? 'Online' : 'Offline'}
          </span>
        </div>
      </header>

      {/* Camera Grid */}
      <div className="camera-grid">
        <CameraPanel
          label="Driver"
          active={isRunning}
          videoSrc={driverVideoSrc || undefined}
          stream={driverStream}
        />
        <CameraPanel
          label="Road"
          active={isRunning}
          videoSrc={roadVideoSrc || undefined}
        />
        <CameraPanel
          label="Cabin"
          active={isRunning}
          videoSrc={cabinVideoSrc || undefined}
        />
      </div>

      {/* Controls */}
      <SessionControls
        mode={mode}
        onModeChange={setMode}
        isRunning={isRunning}
        onStart={handleStart}
        onStop={stopSession}
        driverFile={driverFile}
        roadFile={roadFile}
        cabinFile={cabinFile}
        onDriverFile={setDriverFile}
        onRoadFile={setRoadFile}
        onCabinFile={setCabinFile}
      />

      {/* Voice Indicator — transient, appears only when AI speaks */}
      {voiceMessage && (
        <div className={`voice-indicator ${voiceSeverity === 'CRITICAL' ? 'critical' : 'warning'}`}>
          <SpeakerIcon severity={voiceSeverity} />
          <span className="voice-text">{voiceMessage}</span>
        </div>
      )}
    </div>
  )
}

/* ─── Speaker Icon ─────────────────────────────────────── */
function SpeakerIcon({ severity }: { severity: string }) {
  const color = severity === 'CRITICAL' ? 'var(--accent-red)' :
                severity === 'HIGH' ? 'var(--accent-amber)' : 'var(--accent-blue)'
  return (
    <svg className="voice-icon" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
      <path d="M11 5L6 9H2v6h4l5 4V5z" />
      <path d="M15.54 8.46a5 5 0 010 7.07" />
      <path d="M19.07 4.93a10 10 0 010 14.14" />
    </svg>
  )
}
