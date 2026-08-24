/**
 * SafeRoute AI — API Service
 * Synchronous REST calls to backend. No WebSocket, no history.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface InferResponse {
  warning_message: string | null;
  decision: {
    mode: 'FULL' | 'DEGRADED';
    severity: 'SAFE' | 'CAUTION' | 'HIGH' | 'CRITICAL' | 'DEGRADED';
    action: 'NONE' | 'DROWSINESS_WARNING' | 'DISTRACTION_WARNING' | 'ROAD_WARNING' | 'URGENT_WARNING' | 'DEGRADED_WARNING';
  };
}

export interface HealthResponse {
  status: 'ONLINE' | 'OFFLINE';
  models: Record<string, 'READY' | 'ERROR'>;
}

export const ApiService = {
  /**
   * Check backend health and model status.
   */
  getHealth: async (): Promise<HealthResponse> => {
    try {
      const res = await fetch(`${BASE_URL}/api/health`);
      return res.json();
    } catch {
      return { status: 'OFFLINE', models: {} };
    }
  },

  /**
   * Send up to 3 base64-encoded frames for synchronous inference.
   */
  inferFrames: async (
    driverFrame: string | null,
    roadFrame: string | null,
    cabinFrame: string | null
  ): Promise<InferResponse> => {
    const res = await fetch(`${BASE_URL}/api/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        driver_frame: driverFrame,
        road_frame: roadFrame,
        cabin_frame: cabinFrame,
      }),
    });
    if (!res.ok) throw new Error(`Inference failed: ${res.status}`);
    return res.json();
  },

  /**
   * Upload video files for synchronous batch inference.
   */
  inferVideo: async (
    driverVideo: File | null,
    roadVideo: File | null,
    cabinVideo: File | null
  ): Promise<any> => {
    const formData = new FormData();
    if (driverVideo) formData.append('driver_video', driverVideo);
    if (roadVideo) formData.append('road_video', roadVideo);
    if (cabinVideo) formData.append('cabin_video', cabinVideo);

    const res = await fetch(`${BASE_URL}/api/infer/video`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(`Video inference failed: ${res.status}`);
    return res.json();
  },
};
