interface SessionControlsProps {
  isRunning: boolean
  onStart: () => void
  onStop: () => void
  canStart: boolean
  onExportJson?: () => void
  hasRecording?: boolean
}

/**
 * Session controls — Start/Stop and Export JSON Dataset.
 * Single session start controls all 3 camera channels.
 */
export function SessionControls({
  isRunning, onStart, onStop, canStart, onExportJson, hasRecording
}: SessionControlsProps) {

  return (
    <div className="controls-bar" id="session-controls" style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
      {/* Start / Stop */}
      {isRunning ? (
        <button className="session-btn stop" onClick={onStop} id="btn-stop">
          <StopIcon />
          Stop Session
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

      {/* Export / Download JSON Data */}
      {onExportJson && (
        <button 
          className="session-btn export-json" 
          onClick={onExportJson}
          disabled={isRunning || !hasRecording}
          title="Download all recorded inputs & outputs as JSON"
          style={{
            background: hasRecording && !isRunning ? 'var(--card-bg, #1e293b)' : '#0f172a',
            color: hasRecording && !isRunning ? 'var(--text-primary, #f8fafc)' : '#64748b',
            border: '1px solid #334155',
            padding: '10px 16px',
            borderRadius: '8px',
            cursor: hasRecording && !isRunning ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontWeight: 500,
            fontSize: '14px',
            transition: 'all 0.2s ease'
          }}
        >
          <DownloadIcon />
          Export JSON Data
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

function DownloadIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  )
}
