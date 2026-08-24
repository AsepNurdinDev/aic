/**
 * SafeRoute AI — Voice Alert Service
 * Uses Web Speech Synthesis API for Indonesian TTS.
 * State-transition anti-spam: only plays when action changes or cooldown expires.
 */

class VoiceAlertService {
  private lastAction: string = 'NONE';
  private lastPlayTime: number = 0;
  private cooldownMs: number = 6000;
  private isSpeaking: boolean = false;
  private synthesis: SpeechSynthesis | null = null;

  constructor() {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      this.synthesis = window.speechSynthesis;
    }
  }

  /**
   * Process an inference response and play voice if needed.
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
      this.speak(warningMessage);
      this.lastAction = action;
      this.lastPlayTime = now;
      return true;
    }

    return false;
  }

  /**
   * Speak a message using Web Speech Synthesis.
   */
  private speak(text: string): void {
    if (!this.synthesis) return;

    // Cancel any ongoing speech
    if (this.isSpeaking) {
      this.synthesis.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'id-ID';
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Try to find an Indonesian voice
    const voices = this.synthesis.getVoices();
    const idVoice = voices.find(v => v.lang.startsWith('id'));
    if (idVoice) {
      utterance.voice = idVoice;
    }

    utterance.onstart = () => { this.isSpeaking = true; };
    utterance.onend = () => { this.isSpeaking = false; };
    utterance.onerror = () => { this.isSpeaking = false; };

    this.synthesis.speak(utterance);
  }

  /**
   * Stop any ongoing speech.
   */
  stop(): void {
    if (this.synthesis) {
      this.synthesis.cancel();
      this.isSpeaking = false;
    }
    this.lastAction = 'NONE';
  }

  /**
   * Get current speaking state info for UI indicator.
   */
  getState(): { isSpeaking: boolean; lastAction: string } {
    return {
      isSpeaking: this.isSpeaking,
      lastAction: this.lastAction,
    };
  }
}

export const voiceAlert = new VoiceAlertService();
