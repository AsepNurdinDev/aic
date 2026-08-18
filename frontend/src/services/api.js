/**
 * frontend/src/services/api.js
 * 
 * SafeRoute AI API Client:
 * Menghubungkan frontend React dengan Flask backend AI.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Cek status kesehatan backend & ketersediaan model AI
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn('Backend health check error:', err);
    return { status: 'error', model_ready: false, error: err.message };
  }
}

/**
 * Reset state buffer dan timer kantuk pada backend
 */
export async function resetSession() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    return await res.json();
  } catch (err) {
    console.error('Session reset error:', err);
    return { status: 'error', error: err.message };
  }
}

/**
 * Kirim frame gambar base64 ke backend untuk dianalisis
 * @param {string} base64Image - Data URL / Base64 image
 * @returns {Promise<Object>} Analisis frame lengkap
 */
export async function analyzeFrame(base64Image) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/analyze/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    // Return graceful fallback state
    return {
      face_detected: false,
      status: 'CONNECTION ERROR',
      scenario: 'NO_FACE',
      scenario_label: 'ERROR',
      risk_level: 'LOW',
      risk_score: 0,
      risk_reasons: ['Tidak dapat terhubung ke backend AI'],
      alert_message: 'Pastikan backend Flask berjalan di port 5000.',
      ear: 0.0,
      mar: 0.0,
      pitch_ratio: 1.0,
      yaw_ratio: 0.0,
      head_direction: 'CENTER',
      alarm_active: false,
      fatigue_duration: 0.0,
      target_alert_sec: 4.0,
      error: err.message,
    };
  }
}

/**
 * Upload file video untuk analisis batch/timeline
 * @param {File} driverVideoFile 
 * @returns {Promise<Object>}
 */
export async function analyzeVideoFile(driverVideoFile) {
  const formData = new FormData();
  formData.append('driver_video', driverVideoFile);

  const res = await fetch(`${API_BASE_URL}/api/analyze/video`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `Upload video analysis failed: ${res.statusText}`);
  }

  return await res.json();
}
