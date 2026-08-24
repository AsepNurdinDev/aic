import { useRef, type ChangeEvent, type RefObject } from 'react'

type SessionMode = 'live' | 'demo'

interface SessionControlsProps {
  mode: SessionMode
  onModeChange: (mode: SessionMode) => void
  isRunning: boolean
  onStart: () => void
  onStop: () => void
  driverFile: File | null
  roadFile: File | null
  cabinFile: File | null
  onDriverFile: (f: File | null) => void
  onRoadFile: (f: File | null) => void
  onCabinFile: (f: File | null) => void
}

/**
 * Session controls — mode toggle, start/stop, file inputs for demo mode.
 * Single session start controls all 3 camera channels.
 */
export function SessionControls({
  mode, onModeChange, isRunning, onStart, onStop,
  driverFile, roadFile, cabinFile,
  onDriverFile, onRoadFile, onCabinFile
}: SessionControlsProps) {
  const driverRef = useRef<HTMLInputElement>(null)
  const roadRef = useRef<HTMLInputElement>(null)
  const cabinRef = useRef<HTMLInputElement>(null)

  const handleFile = (setter: (f: File | null) => void) => (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null
    setter(file)
  }

  const canStartDemo = mode === 'demo' && (driverFile || roadFile || cabinFile)
  const canStart = mode === 'live' || canStartDemo

  return (
    <div className="controls-bar" id="session-controls">
      {/* Mode Toggle */}
      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === 'live' ? 'active' : ''}`}
          onClick={() => onModeChange('live')}
          disabled={isRunning}
          id="mode-live"
        >
          Live
        </button>
        <button
          className={`mode-btn ${mode === 'demo' ? 'active' : ''}`}
          onClick={() => onModeChange('demo')}
          disabled={isRunning}
          id="mode-demo"
        >
          Demo
        </button>
      </div>

      {/* File Inputs (Demo mode only) */}
      {mode === 'demo' && !isRunning && (
        <>
          <FileInputButton
            label="Driver"
            file={driverFile}
            inputRef={driverRef}
            onChange={handleFile(onDriverFile)}
            id="file-driver"
          />
          <FileInputButton
            label="Road"
            file={roadFile}
            inputRef={roadRef}
            onChange={handleFile(onRoadFile)}
            id="file-road"
          />
          <FileInputButton
            label="Cabin"
            file={cabinFile}
            inputRef={cabinRef}
            onChange={handleFile(onCabinFile)}
            id="file-cabin"
          />
        </>
      )}

      {/* Start / Stop */}
      {isRunning ? (
        <button className="session-btn stop" onClick={onStop} id="btn-stop">
          <StopIcon />
          Stop
        </button>
      ) : (
        <button
          className="session-btn start"
          onClick={onStart}
          disabled={!canStart}
          id="btn-start"
        >
          <PlayIcon />
          {mode === 'live' ? 'Start Session' : 'Start Demo'}
        </button>
      )}
    </div>
  )
}

/* ─── Sub-components ──────────────────────────────────────── */

function FileInputButton({ label, file, inputRef, onChange, id }: {
  label: string
  file: File | null
  inputRef: RefObject<HTMLInputElement | null>
  onChange: (e: ChangeEvent<HTMLInputElement>) => void
  id: string
}) {
  return (
    <div className="file-input-group">
      <span className="file-label">{label}</span>
      <div className="file-input-wrapper">
        <button
          className={`file-input-btn ${file ? 'has-file' : ''}`}
          onClick={() => inputRef.current?.click()}
          id={id}
        >
          {file ? file.name : 'Choose file'}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="file-input-hidden"
          onChange={onChange}
        />
      </div>
    </div>
  )
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5v14l11-7z" />
    </svg>
  )
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  )
}
