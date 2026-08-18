"""
src/inference.py

Module untuk menjalankan inferensi model LandmarkGRU yang telah dilatih:
- Input sequence: tensor bentuk (1, 60, 2)
- Output: Probabilitas kelas (NOT DROWSY / DROWSY) dan label prediksi

Checkpoint: models/best_landmark_gru_ear_mar.pth
"""

from typing import Dict, Optional, Union
import torch
import torch.nn.functional as F

from src.model import load_model
from src.preprocessing import SequenceBuffer


class DrowsinessInference:
    """
    Inference engine untuk deteksi kantuk dari sequence EAR + MAR.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/best_landmark_gru_ear_mar.pth",
        config_path: Optional[str] = "models/FINAL_landmark_gru_config.json",
        device: str = "cpu",
        default_threshold: float = 0.50
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.default_threshold = default_threshold

        # Load model and verify checkpoint
        self.model, self.metadata = load_model(
            checkpoint_path=self.checkpoint_path,
            device=str(self.device),
            config_path=self.config_path
        )
        self.model.eval()

        self.class_names = {
            0: "NOT DROWSY",
            1: "DROWSY"
        }

    def predict(
        self,
        sequence: Union[torch.Tensor, SequenceBuffer],
        threshold: Optional[float] = None
    ) -> Dict[str, Union[float, int, str]]:
        """
        Jalankan inferensi pada sequence ternormalisasi.

        Args:
            sequence: Tensor bentuk (1, 60, 2) atau (60, 2), atau instance SequenceBuffer.
            threshold: Optional custom decision threshold untuk kelas DROWSY (default: 0.50).

        Returns:
            Dict:
                - "not_drowsy_probability": float
                - "drowsy_probability": float
                - "prediction": int (0 atau 1)
                - "label": str ("NOT DROWSY" atau "DROWSY")
        """
        thresh = threshold if threshold is not None else self.default_threshold

        # Handle SequenceBuffer input directly
        if hasattr(sequence, "get_sequence"):
            sequence_tensor = sequence.get_sequence(normalize=True)
        elif isinstance(sequence, torch.Tensor):
            sequence_tensor = sequence
        else:
            raise TypeError(f"Expected torch.Tensor or SequenceBuffer, got {type(sequence)}")

        # Ensure correct dimension: (60, 2) -> (1, 60, 2)
        if sequence_tensor.ndim == 2:
            sequence_tensor = sequence_tensor.unsqueeze(0)
        elif sequence_tensor.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D tensor shape (1, 60, 2), got shape: {tuple(sequence_tensor.shape)}"
            )

        sequence_tensor = sequence_tensor.to(self.device).float()

        # Sanity check input
        if torch.isnan(sequence_tensor).any() or torch.isinf(sequence_tensor).any():
            raise ValueError("Input sequence tensor contains NaN or Inf.")

        # Inference forward pass
        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probs = F.softmax(logits, dim=-1)[0]

        not_drowsy_prob = float(probs[0].item())
        drowsy_prob = float(probs[1].item())

        # Sanity checks on output probabilities
        if not (0.0 <= not_drowsy_prob <= 1.0) or not (0.0 <= drowsy_prob <= 1.0):
            raise ValueError(f"Probabilities out of bounds: [{not_drowsy_prob}, {drowsy_prob}]")

        prob_sum = not_drowsy_prob + drowsy_prob
        if abs(prob_sum - 1.0) > 1e-4:
            raise ValueError(f"Probabilities do not sum to 1.0: sum = {prob_sum}")

        # Classification decision based on threshold
        if drowsy_prob >= thresh:
            pred_class = 1
        else:
            pred_class = 0

        return {
            "not_drowsy_probability": not_drowsy_prob,
            "drowsy_probability": drowsy_prob,
            "prediction": pred_class,
            "label": self.class_names[pred_class]
        }
