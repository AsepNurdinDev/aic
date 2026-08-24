import { useRef, useEffect, ReactNode } from 'react'

interface CameraPanelProps {
  label: string
  active: boolean
  videoSrc?: string           // Object URL for demo video
  stream?: MediaStream | null // MediaStream for live camera
  selector?: ReactNode        // Dropdown or file input
  id?: string
}

/**
 * Minimal camera panel — shows live feed or uploaded video.
 * Label above the camera, LIVE and timestamp inside the video.
 */
export function CameraPanel({ label, active, videoSrc, stream, selector, id }: CameraPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null)

  // Attach MediaStream when in live mode
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream
    }
    return () => {
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    }
  }, [stream])

  const hasSource = !!videoSrc || !!stream

  // Current formatted time for the overlay
  const now = new Date()
  const dateStr = now.toISOString().split('T')[0]
  const timeStr = now.toTimeString().split(' ')[0]

  return (
    <div className="camera-panel-container" id={id}>
      <div className="camera-header">
        <span className="camera-label">{label}</span>
        {selector}
      </div>
      
      <div className={`camera-panel ${active ? 'active' : ''}`}>
        {(active && hasSource) && (
          <div className="live-indicator">
            <span className="live-dot" />
            LIVE
          </div>
        )}

        {(active && hasSource) && (
          <div className="timestamp-indicator">
            {dateStr} {timeStr}
          </div>
        )}

        {hasSource ? (
          <video
            ref={videoRef}
            className="camera-video"
            src={videoSrc || undefined}
            autoPlay
            muted
            playsInline
            loop={!!videoSrc}
          />
        ) : (
          <div className="camera-placeholder">
            <svg className="camera-placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
            </svg>
            <span className="camera-placeholder-text">No feed</span>
          </div>
        )}
      </div>
    </div>
  )
}
