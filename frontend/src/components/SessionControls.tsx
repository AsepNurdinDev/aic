interface SessionControlsProps {
  isRunning: boolean
  onStart: () => void
  onStop: () => void
  canStart: boolean
}

/**
 * Session controls — just Start/Stop.
 * Single session start controls all 3 camera channels.
 */
export function SessionControls({
  isRunning, onStart, onStop, canStart
}: SessionControlsProps) {

  return (
    <div className="controls-bar" id="session-controls">
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
          Start Session
        </button>
      )}
    </div>
  )
}

/* ─── Sub-components ──────────────────────────────────────── */

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
