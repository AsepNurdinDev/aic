/**
 * frontend/src/services/audioAlert.js
 * 
 * Audio Alert Manager menggunakan Web Audio API:
 * Menghasilkan bunyi alarm peringatan kantuk secara non-blocking di browser.
 */

class AudioAlertService {
  constructor() {
    this.audioCtx = null;
    this.isPlaying = false;
    this.intervalId = null;
    this.frequency = 1500; // Hz
    this.pulseDuration = 0.35; // seconds
    this.repeatInterval = 600; // ms
  }

  _getAudioContext() {
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        this.audioCtx = new AudioContextClass();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  playSingleBeep() {
    try {
      const ctx = this._getAudioContext();
      if (!ctx) return;

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(this.frequency, ctx.currentTime);

      // Volume envelope (smooth attack and decay to avoid clicks)
      gain.gain.setValueAtTime(0.01, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + this.pulseDuration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + this.pulseDuration);
    } catch (e) {
      console.warn('Audio alert playback error:', e);
    }
  }

  startAlarm() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.playSingleBeep();
    this.intervalId = setInterval(() => {
      this.playSingleBeep();
    }, this.repeatInterval);
  }

  stopAlarm() {
    if (!this.isPlaying) return;
    this.isPlaying = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
}

export const audioAlert = new AudioAlertService();
