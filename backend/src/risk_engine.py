"""
backend/src/risk_engine.py

SafeRoute AI Contextual Risk Engine & Explainability Module:
Menghitung tingkat risiko (LOW, POTENTIAL, WARNING, CRITICAL), skor risiko (0-100),
dan alasan kontekstual (AI explainability) berdasarkan kondisi pengemudi & lingkungan jalan.
"""

from typing import Dict, List, Any, Optional


class RiskEngine:
    """
    Kalkulator risiko kontekstual SafeRoute AI.
    """

    def evaluate_risk(
        self,
        scenario: str,
        fatigue_duration: float,
        target_alert_sec: float,
        alarm_active: bool,
        smoothed_drowsy_prob: Optional[float],
        is_nodding: bool,
        is_looking_aside: bool,
        head_direction: str,
        ear: float,
        mar: float,
        road_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Menghasilkan skor risiko, tingkat risiko, dan alasan kontekstual.
        """
        risk_score = 0
        reasons = []
        alert_message = ""

        # 1. Evaluasi Kondisi Pengemudi
        if scenario == "MICROSLEEP":
            if alarm_active:
                risk_score += 95
                reasons.append(f"Mata terpejam / microsleep kritis ({fatigue_duration:.1f}s >= {target_alert_sec:.1f}s)")
                alert_message = "BAHAYA! Mata terpejam, segera fokus ke jalan!"
            else:
                risk_score += 70
                reasons.append(f"Indikasi mata terpejam ({fatigue_duration:.1f}s)")
                alert_message = "Peringatan: Buka mata Anda dan perhatikan jalan."

        elif scenario == "NODDING":
            if alarm_active:
                risk_score += 85
                reasons.append(f"Kepala terkulai / nodding ({fatigue_duration:.1f}s >= {target_alert_sec:.1f}s)")
                alert_message = "PERINGATAN! Kepala menunduk, tegakkan posisi duduk!"
            else:
                risk_score += 60
                reasons.append(f"Posisi kepala menunduk ({fatigue_duration:.1f}s)")
                alert_message = "Perhatikan jalan di depan."

        elif scenario == "LOOK_ASIDE":
            if alarm_active:
                risk_score += 80
                reasons.append(f"Distraksi: Menengok ke arah {head_direction} ({fatigue_duration:.1f}s >= {target_alert_sec:.1f}s)")
                alert_message = f"PERHATIAN! Arahkan pandangan ke depan (menengok {head_direction})!"
            else:
                risk_score += 45
                reasons.append(f"Pandangan mengarah ke {head_direction} ({fatigue_duration:.1f}s)")
                alert_message = "Fokuskan pandangan ke arah jalan."

        elif scenario == "DROWSY":
            prob_percent = int((smoothed_drowsy_prob or 0.5) * 100)
            if alarm_active:
                risk_score += 85
                reasons.append(f"Pola kantuk persisten terdeteksi ({prob_percent}%, durasi {fatigue_duration:.1f}s)")
                alert_message = "PERINGATAN! Anda menunjukkan gejala kantuk berat, istirahat jika perlu."
            else:
                risk_score += 55
                reasons.append(f"Tanda-tanda kantuk terdeteksi ({prob_percent}%)")
                alert_message = "Tetap waspada dan jaga konsentrasi berkendara."

        elif scenario == "YAWNING":
            if alarm_active:
                risk_score += 50
                reasons.append(f"Menguap berulang / durasi panjang ({fatigue_duration:.1f}s)")
                alert_message = "Indikasi kelelahan awal terdeteksi."
            else:
                risk_score += 30
                reasons.append("Pengemudi sedang menguap (kelelahan awal)")
                alert_message = "Peringatan dini: Indikasi lelah."

        elif scenario == "BUFFERING":
            risk_score = 10
            reasons.append("Sedang mengumpulkan sequence data untuk inferensi...")
            alert_message = "Menginisialisasi deteksi..."

        elif scenario == "NO_FACE":
            risk_score = 25
            reasons.append("Wajah pengemudi tidak terdeteksi dalam frame")
            alert_message = "Pastikan wajah pengemudi terlihat jelas oleh kamera."

        else:  # NORMAL / ALERT
            risk_score = 10
            reasons.append("Pengemudi sadar dan fokus ke arah jalan")
            alert_message = "Kondisi normal dan aman."

        # 2. Evaluasi Kondisi Jalan (jika ada context tambahan)
        if road_context:
            if road_context.get("vehicle_close"):
                risk_score += 20
                reasons.append("Kendaraan depan dalam jarak dekat")
            if road_context.get("heavy_traffic"):
                risk_score += 10
                reasons.append("Kepadatan lalu lintas tinggi")

        # Batasi skor 0 - 100
        risk_score = max(0, min(100, risk_score))

        # 3. Tentukan Risk Level
        if risk_score >= 75:
            risk_level = "CRITICAL"
        elif risk_score >= 45:
            risk_level = "WARNING"
        elif risk_score >= 25:
            risk_level = "POTENTIAL"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_reasons": reasons,
            "alert_message": alert_message
        }
