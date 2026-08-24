import { useState, useRef, useCallback, useEffect } from 'react'
import { CameraPanel } from '../components/CameraPanel'
import { SessionControls } from '../components/SessionControls'
import { ApiService, InferResponse } from '../services/api'
import { voiceAlert } from '../services/voiceAlert'

type PanelState = {
  sourceType: 'none' | 'file' | 'device'
  deviceId?: string
  file?: File
  url?: string
  stream?: MediaStream
}

/**
 * MonitorPage — Unified live & demo processing.
 */
export function MonitorPage() {
  // ─── State ──────────────────────────────────────────────
  const [isRunning, setIsRunning] = useState(false)
  const [backendOnline, setBackendOnline] = useState(false)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])

  const [driverState, setDriverState] = useState<PanelState>({ sourceType: 'none' })
  const [roadState, setRoadState] = useState<PanelState>({ sourceType: 'none' })
  const [cabinState, setCabinState] = useState<PanelState>({ sourceType: 'none' })

  // Voice indicator
  const [voiceMessage, setVoiceMessage] = useState<string | null>(null)
  const [voiceSeverity, setVoiceSeverity] = useState<string>('SAFE')
  const [hasRecording, setHasRecording] = useState(false)
  const [latestFileName, setLatestFileName] = useState<string | null>(null)

  // Refs for cleanup
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const voiceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ─── Init Devices & Health ──────────────────────────────
  useEffect(() => {
    const checkHealth = async () => {
      const health = await ApiService.getHealth()
      setBackendOnline(health.status === 'ONLINE')
    }
    checkHealth()
    const hInterval = setInterval(checkHealth, 10000)

    const initDevices = async () => {
      try {
        // Request permission to get labels
        await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        const devs = await navigator.mediaDevices.enumerateDevices()
        setDevices(devs.filter(d => d.kind === 'videoinput'))
      } catch (err) {
        console.warn('Could not enumerate devices', err)
      }
    }
    initDevices()

    return () => clearInterval(hInterval)
  }, [])

  // ─── Source Selection Logic ─────────────────────────────
  const cleanupPanel = (state: PanelState) => {
    if (state.stream) state.stream.getTracks().forEach(t => t.stop())
    if (state.url) URL.revokeObjectURL(state.url)
  }

  const handleDeviceSelect = async (deviceId: string, setState: (s: PanelState) => void, oldState: PanelState) => {
    cleanupPanel(oldState)
    if (deviceId === 'none') {
      setState({ sourceType: 'none' })
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId } } })
      setState({ sourceType: 'device', deviceId, stream })
    } catch (err) {
      console.error('Failed to start device', err)
      setState({ sourceType: 'none' })
    }
  }

  const handleFileSelect = (file: File, setState: (s: PanelState) => void, oldState: PanelState) => {
    cleanupPanel(oldState)
    setState({ sourceType: 'file', file, url: URL.createObjectURL(file) })
  }

  // ─── Capture frame from video element ───────────────────
  const captureFrame = useCallback((videoId: string): string | null => {
    const videoEl = document.querySelector(`#${videoId} video`) as HTMLVideoElement
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

  // ─── Start Session ──────────────────────────────────────
  const startSession = useCallback(() => {
    setIsRunning(true)
    setHasRecording(false)

    // Notify backend to start recording dataset
    ApiService.startRecording().catch(() => {})

    // Ensure all file videos are playing (they might be paused)
    document.querySelectorAll('.camera-video').forEach((el) => {
      const vid = el as HTMLVideoElement
      vid.play().catch(() => {})
    })

    // Send frames at ~3 FPS
    intervalRef.current = setInterval(async () => {
      const driverFrame = captureFrame('panel-driver')
      const roadFrame = captureFrame('panel-road')
      const cabinFrame = captureFrame('panel-cabin')

      if (!driverFrame && !roadFrame && !cabinFrame) return

      try {
        const res = await ApiService.inferFrames(driverFrame, roadFrame, cabinFrame)
        handleInferResponse(res)
      } catch {
        // Silently ignore inference errors to keep streaming
      }
    }, 333)
  }, [captureFrame, handleInferResponse])

  // ─── Stop Session ───────────────────────────────────────
  const stopSession = useCallback(async () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    // Pause all videos
    document.querySelectorAll('.camera-video').forEach((el) => {
      const vid = el as HTMLVideoElement
      vid.pause()
    })

    voiceAlert.stop()
    setVoiceMessage(null)
    setIsRunning(false)

    // Stop recording on backend and save JSON dataset
    try {
      const res = await ApiService.stopRecording()
      if (res && res.status === 'SAVED') {
        setHasRecording(true)
        if (res.filename) setLatestFileName(res.filename)
      }
    } catch (err) {
      console.warn('Error saving recording:', err)
    }
  }, [])

  // ─── Export / Download JSON Data ────────────────────────
  const handleExportJson = useCallback(async () => {
    try {
      let data
      if (latestFileName) {
        window.open(ApiService.getDownloadUrl(latestFileName), '_blank')
        return
      }
      data = await ApiService.getLatestRecording()
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `session_record_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '_')}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      alert('Could not export recording JSON: ' + String(err))
    }
  }, [latestFileName])

  // ─── Cleanup on unmount ─────────────────────────────────
  useEffect(() => {
    return () => {
      stopSession()
      cleanupPanel(driverState)
      cleanupPanel(roadState)
      cleanupPanel(cabinState)
    }
  }, []) // empty dep array is fine for unmount cleanup in this case (uses ref for stopSession)

  const canStart = driverState.sourceType !== 'none' || roadState.sourceType !== 'none' || cabinState.sourceType !== 'none'

  // ─── Render ─────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">SR</div>
          <h1 className="header-title">FleetSense AI</h1>
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
          id="panel-driver"
          label="Driver"
          active={isRunning}
          videoSrc={driverState.url}
          stream={driverState.stream}
          selector={
            <SourceSelector 
              value={driverState.sourceType === 'device' ? (driverState.deviceId || 'none') : (driverState.sourceType === 'file' ? 'file' : 'none')}
              devices={devices}
              disabled={isRunning}
              onSelectDevice={(id) => handleDeviceSelect(id, setDriverState, driverState)}
              onSelectFile={(f) => handleFileSelect(f, setDriverState, driverState)}
            />
          }
        />
        <CameraPanel
          id="panel-road"
          label="Road"
          active={isRunning}
          videoSrc={roadState.url}
          stream={roadState.stream}
          selector={
            <SourceSelector 
              value={roadState.sourceType === 'device' ? (roadState.deviceId || 'none') : (roadState.sourceType === 'file' ? 'file' : 'none')}
              devices={devices}
              disabled={isRunning}
              onSelectDevice={(id) => handleDeviceSelect(id, setRoadState, roadState)}
              onSelectFile={(f) => handleFileSelect(f, setRoadState, roadState)}
            />
          }
        />
        <CameraPanel
          id="panel-cabin"
          label="Cabin"
          active={isRunning}
          videoSrc={cabinState.url}
          stream={cabinState.stream}
          selector={
            <SourceSelector 
              value={cabinState.sourceType === 'device' ? (cabinState.deviceId || 'none') : (cabinState.sourceType === 'file' ? 'file' : 'none')}
              devices={devices}
              disabled={isRunning}
              onSelectDevice={(id) => handleDeviceSelect(id, setCabinState, cabinState)}
              onSelectFile={(f) => handleFileSelect(f, setCabinState, cabinState)}
            />
          }
        />
      </div>

      {/* Controls */}
      <SessionControls
        isRunning={isRunning}
        canStart={canStart}
        onStart={startSession}
        onStop={stopSession}
        onExportJson={handleExportJson}
        hasRecording={hasRecording}
      />

      {/* Voice Indicator */}
      {voiceMessage && (
        <div className={`voice-indicator ${voiceSeverity === 'CRITICAL' ? 'critical' : 'warning'}`}>
          <SpeakerIcon severity={voiceSeverity} />
          <span className="voice-text">{voiceMessage}</span>
        </div>
      )}
    </div>
  )
}

/* ─── Source Selector Component ────────────────────────── */
function SourceSelector({ 
  value, 
  devices,
  disabled,
  onSelectDevice,
  onSelectFile
}: {
  value: string
  devices: MediaDeviceInfo[]
  disabled: boolean
  onSelectDevice: (deviceId: string) => void
  onSelectFile: (file: File) => void
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value
    if (val === 'file') {
      fileInputRef.current?.click()
    } else {
      onSelectDevice(val)
    }
  }
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onSelectFile(file)
    }
    // reset input so the same file can be selected again if needed
    e.target.value = ''
  }

  return (
    <>
      <select className="camera-source-select" value={value} onChange={handleChange} disabled={disabled}>
        <option value="none">None</option>
        <option value="file">File Video...</option>
        {devices.map(d => (
          <option key={d.deviceId} value={d.deviceId}>
            {d.label || `Camera ${d.deviceId.substring(0, 5)}`}
          </option>
        ))}
      </select>
      <input 
        type="file" 
        accept="video/*" 
        ref={fileInputRef} 
        className="file-input-hidden" 
        onChange={handleFileChange} 
      />
    </>
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
