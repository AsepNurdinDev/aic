"""
src/smoothing.py

Temporal smoothing untuk menstabilkan prediksi probabilitas kantuk secara realtime.
Menggunakan moving average dengan window geser masa lalu (causal filter tanpa future frames).
"""

from collections import deque
from typing import Dict, Union, Optional


class PredictionSmoother:
    """
    Penghalus probabilitas berbasis sliding window moving average.
    """

    def __init__(
        self,
        window_size: int = 5,
        threshold: float = 0.50
    ):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

        self.window_size = window_size
        self.threshold = threshold
        self.history = deque(maxlen=self.window_size)
        self.class_names = {
            0: "NOT DROWSY",
            1: "DROWSY"
        }

    def update(
        self,
        drowsy_probability: float,
        threshold: Optional[float] = None
    ) -> Dict[str, Union[float, int, str]]:
        """
        Tambahkan probabilitas kantuk frame saat ini ke riwayat dan hitung moving average.

        Args:
            drowsy_probability: Probabilitas kelas DROWSY dari model (rentang 0.0 - 1.0).
            threshold: Optional threshold override.

        Returns:
            Dict:
                - "raw_probability": float
                - "smoothed_probability": float
                - "prediction": int (0 atau 1)
                - "label": str ("NOT DROWSY" atau "DROWSY")
        """
        thresh = threshold if threshold is not None else self.threshold

        raw_prob = float(drowsy_probability)
        self.history.append(raw_prob)

        # Hitung mean dari window riwayat saat ini
        smoothed_prob = float(sum(self.history) / len(self.history))

        # Keputusan berdasarkan smoothed probability
        if smoothed_prob >= thresh:
            pred = 1
        else:
            pred = 0

        return {
            "raw_probability": raw_prob,
            "smoothed_probability": smoothed_prob,
            "prediction": pred,
            "label": self.class_names[pred]
        }

    def reset(self):
        """Reset riwayat window smoothing."""
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)
