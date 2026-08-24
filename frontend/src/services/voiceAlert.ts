/**
 * SafeRoute AI — Local Audio & Voice Alert Service
 * Memutar file audio lokal (.mp3) berdasarkan action yang dideteksi.
 */

class VoiceAlertService {
  private lastAction: string = 'NONE';
  private lastPlayTime: number = 0;
  private cooldownMs: number = 6000;
  private currentAudio: HTMLAudioElement | null = null;

  // Mapping action backend ke file audio lokal di folder /public/audio/
  private audioMap: Record<string, string> = {
    'DROWSINESS_WARNING': '/audio/drowsiness.mp3',
    'DISTRACTION_WARNING': '/audio/distraction.mp3',
    'ROAD_WARNING': '/audio/road_warning.mp3',
    'URGENT_WARNING': '/audio/urgent.mp3',
    'DEGRADED_WARNING': '/audio/warning.mp3',
  };

  /**
   * Proses respon inferensi dan putar audio lokal.
   * Returns true if voice was played.
   */
  processResponse(warningMessage: string | null, action: string): boolean {
    if (!warningMessage || action === 'NONE') {
      this.lastAction = action;
      return false;
    }

    const now = Date.now();
    const actionChanged = action !== this.lastAction;
    const cooldownExpired = (now - this.lastPlayTime) > this.cooldownMs;

    if (actionChanged || cooldownExpired) {
      this.playLocalAudio(action, warningMessage);
      this.lastAction = action;
      this.lastPlayTime = now;
      return true;
    }

    return false;
  }

  /**
   * Putar file MP3 lokal. Jika file tidak ada, fallback ke Web Speech API.
   */
  private playLocalAudio(action: string, fallbackText: string): void {
    // Hentikan audio yang sedang berputar jika ada
    this.stop();

    const audioSrc = this.audioMap[action];

    if (audioSrc) {
      const audio = new Audio(audioSrc);
      this.currentAudio = audio;

      audio.play().catch((err) => {
        console.warn(`[VoiceAlert] Gagal memutar audio lokal (${audioSrc}), fallback ke TTS:`, err);
        this.speakFallback(fallbackText);
      });
    } else {
      this.speakFallback(fallbackText);
    }
  }

  /**
   * Fallback Web Speech Synthesis jika file audio belum tersedia
   */
  private speakFallback(text: string): void {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'id-ID';
    window.speechSynthesis.speak(utterance);
  }

  /**
   * Hentikan audio yang sedang diputar.
   */
  stop(): void {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    this.lastAction = 'NONE';
  }

  /**
   * Get current speaking state info for UI indicator.
   */
  getState(): { isSpeaking: boolean; lastAction: string } {
    const isSpeaking = this.currentAudio !== null ? !this.currentAudio.paused : false;
    return {
      isSpeaking: isSpeaking,
      lastAction: this.lastAction,
    };
  }
}

export const voiceAlert = new VoiceAlertService();
