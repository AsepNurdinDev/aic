/**
 * SafeRoute AI — Local Audio & Voice Alert Service
 * Memutar file audio lokal (.mp3) berdasarkan action yang dideteksi.
 */

class VoiceAlertService {
  private lastAction: string = 'NONE';

  // Cooldown antar peringatan yang SAMA (jeda sebelum peringatan yang sama boleh bunyi lagi)
  private actionCooldowns: Record<string, number> = {
    'DROWSINESS_WARNING': 10000, // 10 detik jeda untuk drowsiness
    'ROAD_WARNING': 8000,        // 8 detik
    'URGENT_WARNING': 5000,      // 5 detik (lebih sering karena urgent)
    'DEGRADED_WARNING': 15000,   // 15 detik
    'DEFAULT': 6000
  };

  // Delay/waktu tunggu sebelum peringatan diputar sejak pertama kali terdeteksi (0 ms = langsung bunyi)
  private actionDelays: Record<string, number> = {
    'DROWSINESS_WARNING': 0, // Langsung bunyi tanpa delay
    'ROAD_WARNING': 0,       // Langsung bunyi tanpa delay
    'URGENT_WARNING': 0,     // Langsung bunyi tanpa delay
    'DEGRADED_WARNING': 0,   // Langsung bunyi tanpa delay
    'DEFAULT': 0             // Langsung bunyi tanpa delay
  };

  private lastPlayTimes: Record<string, number> = {};
  private conditionStartTimes: Record<string, number> = {};
  
  // Cooldown global agar peringatan yang berbeda tidak berbunyi bersamaan atau berurutan cepat
  private globalCooldownMs: number = 3000; 
  private lastGlobalPlayTime: number = 0;

  private currentAudio: HTMLAudioElement | null = null;

  // Mapping action backend ke file audio lokal di folder /public/audio/
  private audioMap: Record<string, string> = {
    'DROWSINESS_WARNING': '/audio/drowsiness.mp3',
    // 'DISTRACTION_WARNING': '/audio/distraction.mp3',
    'ROAD_WARNING': '/audio/road_warning.mp3',
    'URGENT_WARNING': '/audio/urgent.mp3',
    'DEGRADED_WARNING': '/audio/warning.mp3',
  };

  /**
   * Proses respon inferensi dan putar audio lokal.
   * Returns true if voice was played.
   */
  processResponse(warningMessage: string | null, action: string): boolean {
    const now = Date.now();

    if (!warningMessage || action === 'NONE') {
      if (this.lastAction !== 'NONE') {
        this.lastAction = 'NONE';
      }
      return false;
    }

    // Jika deteksi kondisi baru (atau berubah dari sebelumnya)
    if (action !== this.lastAction) {
      this.lastAction = action;
      // Hanya reset waktu mulai jika sebelumnya belum ada catatan untuk aksi ini
      // atau jika sudah lama tidak terdeteksi (misal lebih dari 5 detik)
      const lastStart = this.conditionStartTimes[action] || 0;
      if (now - lastStart > 5000) {
        this.conditionStartTimes[action] = now;
      }
    }

    // 1. Cek Delay (apakah kondisi sudah bertahan cukup lama?)
    const requiredDelay = this.actionDelays[action] ?? this.actionDelays['DEFAULT'];
    const startTime = this.conditionStartTimes[action] || now;
    const hasPassedDelay = (now - startTime) >= requiredDelay;

    // 2. Cek Cooldown untuk Aksi ini (apakah sudah cukup lama sejak terakhir kali bunyi?)
    const actionCooldown = this.actionCooldowns[action] ?? this.actionCooldowns['DEFAULT'];
    const lastPlayed = this.lastPlayTimes[action] || 0;
    const hasPassedActionCooldown = (now - lastPlayed) > actionCooldown;

    // 3. Cek Global Cooldown (apakah tidak bertumpuk dengan peringatan lain?)
    const hasPassedGlobalCooldown = (now - this.lastGlobalPlayTime) > this.globalCooldownMs;

    if (hasPassedDelay && hasPassedActionCooldown && hasPassedGlobalCooldown) {
      this.playLocalAudio(action, warningMessage);
      
      this.lastPlayTimes[action] = now;
      this.lastGlobalPlayTime = now;
      
      // Update start time agar jika kondisinya terus bertahan, akan menunggu cooldown
      this.conditionStartTimes[action] = now;
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
