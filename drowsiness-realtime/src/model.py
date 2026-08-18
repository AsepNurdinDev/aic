"""
src/model.py

Definisi arsitektur LandmarkGRU yang identik dengan model training:
- Input: [EAR, MAR] (input_dim = 2)
- Normalization: nn.LayerNorm(input_dim)
- GRU: 2-layer GRU, hidden_size = 128, batch_first = True
- Attention: Temporal Attention (score: Linear(128, 64) -> Tanh -> Linear(64, 1))
- Classifier: Dropout -> Linear(128, 64) -> ReLU -> Dropout -> Linear(64, 2)
- Output: 2 classes (0 = NOT DROWSY, 1 = DROWSY)

Checkpoint: models/best_landmark_gru_ear_mar.pth
"""

import os
import json
from typing import Optional, Tuple
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Temporal Attention layer to compute context vector over sequence timestamps.
    Matches the exact attention module in the trained checkpoint.
    """

    def __init__(self, hidden_dim: int = 128, attention_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, gru_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_out: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            context: Tensor of shape (batch_size, hidden_dim)
            weights: Tensor of shape (batch_size, seq_len, 1)
        """
        # scores shape: (batch_size, seq_len, 1)
        scores = self.score(gru_out)
        weights = torch.softmax(scores, dim=1)
        # context shape: (batch_size, hidden_dim)
        context = torch.sum(gru_out * weights, dim=1)
        return context, weights


class LandmarkGRU(nn.Module):
    """
    LandmarkGRU architecture for drowsiness classification from temporal EAR/MAR features.
    Architecture is strictly identical to the trained checkpoint best_landmark_gru_ear_mar.pth.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Normalization layer on input features (shape: [input_dim])
        self.norm = nn.LayerNorm(input_dim)

        # 2-layer GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Temporal attention mechanism
        self.attention = TemporalAttention(hidden_dim=hidden_dim, attention_dim=64)

        # Classification head matching checkpoint structure
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Output logits of shape (batch_size, num_classes)
        """
        x = self.norm(x)
        gru_out, _ = self.gru(x)
        context, _ = self.attention(gru_out)
        logits = self.classifier(context)
        return logits


def create_model(
    input_dim: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.2
) -> LandmarkGRU:
    """
    Helper function to instantiate LandmarkGRU.
    """
    return LandmarkGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    )


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    config_path: Optional[str] = None
) -> Tuple[LandmarkGRU, dict]:
    """
    Load LandmarkGRU model and trained weights from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Torch device ("cpu" or "cuda").
        config_path: Optional path to configuration JSON.

    Returns:
        model: LandmarkGRU in eval mode on target device.
        metadata: Dictionary containing training stats, config, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load configuration if provided
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint file {checkpoint_path}: {e}")

    # Extract state dict and parameters
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "macro_f1": checkpoint.get("macro_f1"),
            "feature_columns": checkpoint.get("feature_columns", ["ear", "mar"]),
            "sequence_length": checkpoint.get("sequence_length", 60),
            "feature_mean": checkpoint.get("feature_mean", [0.32720324397087097, 0.2745433449745178]),
            "feature_std": checkpoint.get("feature_std", [0.08445222675800323, 0.1241738498210907]),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")

    input_dim = config.get("input_dim", 2)
    hidden_dim = config.get("hidden_dim", 128)
    num_layers = config.get("gru_layers", 2)
    num_classes = 2

    # Instantiate model
    model = create_model(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # Load weights with strict matching
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"State dict mismatch when loading {checkpoint_path} into LandmarkGRU: {e}"
        )

    model.to(device)
    model.eval()

    return model, metadata
