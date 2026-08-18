"""
src/preprocessing.py

Preprocessing and buffering for EAR + MAR sequence data:
- Z-score normalization using TRAINING mean and std (from checkpoint/config)
- Buffer maintaining sliding window of 60 frames
- Conversion to torch.FloatTensor with shape (1, 60, 2)

Formula:
  normalized = (feature - train_mean) / train_std

DO NOT calculate mean or std from webcam.
DO NOT use Relative-EAR, PERCLOS, or delta features.
"""

import os
import json
from collections import deque
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

# Default training statistics from best_landmark_gru_ear_mar.pth metadata
DEFAULT_FEATURE_MEAN: List[float] = [0.32720324397087097, 0.2745433449745178]
DEFAULT_FEATURE_STD: List[float] = [0.08445222675800323, 0.1241738498210907]
DEFAULT_SEQUENCE_LENGTH: int = 60


def load_training_statistics(
    checkpoint_path: Optional[str] = "models/best_landmark_gru_ear_mar.pth",
    config_path: Optional[str] = "models/FINAL_landmark_gru_config.json"
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load feature mean, feature std, and sequence length from checkpoint or config file.

    Returns:
        mean: np.ndarray shape (2,)
        std: np.ndarray shape (2,)
        seq_len: int
    """
    mean = np.array(DEFAULT_FEATURE_MEAN, dtype=np.float32)
    std = np.array(DEFAULT_FEATURE_STD, dtype=np.float32)
    seq_len = DEFAULT_SEQUENCE_LENGTH

    # Try loading from checkpoint file first (most accurate metadata)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "feature_mean" in ckpt and ckpt["feature_mean"] is not None:
                    mean = np.array(ckpt["feature_mean"], dtype=np.float32)
                if "feature_std" in ckpt and ckpt["feature_std"] is not None:
                    std = np.array(ckpt["feature_std"], dtype=np.float32)
                if "sequence_length" in ckpt and ckpt["sequence_length"] is not None:
                    seq_len = int(ckpt["sequence_length"])
                return mean, std, seq_len
        except Exception:
            pass

    # Try loading from config JSON file
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "sequence_length" in cfg:
                    seq_len = int(cfg["sequence_length"])
                if "feature_mean" in cfg:
                    mean = np.array(cfg["feature_mean"], dtype=np.float32)
                if "feature_std" in cfg:
                    std = np.array(cfg["feature_std"], dtype=np.float32)
        except Exception:
            pass

    return mean, std, seq_len


class SequenceBuffer:
    """
    Circular sliding-window buffer storing sequence of [EAR, MAR] features.
    Provides normalized sequence tensor of shape (1, 60, 2) for model inference.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        feature_mean: Optional[Union[List[float], np.ndarray]] = None,
        feature_std: Optional[Union[List[float], np.ndarray]] = None
    ):
        self.sequence_length = sequence_length

        if feature_mean is not None:
            self.mean = np.array(feature_mean, dtype=np.float32)
        else:
            self.mean = np.array(DEFAULT_FEATURE_MEAN, dtype=np.float32)

        if feature_std is not None:
            self.std = np.array(feature_std, dtype=np.float32)
        else:
            self.std = np.array(DEFAULT_FEATURE_STD, dtype=np.float32)

        # Prevent division by zero
        self.std = np.where(self.std < 1e-7, 1e-7, self.std)

        # Deque for fast sliding window storage
        self.buffer = deque(maxlen=self.sequence_length)

    def append(self, feature: Union[List[float], Tuple[float, float], np.ndarray]):
        """
        Append a single frame's [EAR, MAR] feature vector.
        """
        if isinstance(feature, (list, tuple)):
            if len(feature) < 2:
                raise ValueError(f"Feature must contain at least 2 elements [EAR, MAR], got: {feature}")
            val = [float(feature[0]), float(feature[1])]
        elif isinstance(feature, np.ndarray):
            val = [float(feature[0]), float(feature[1])]
        else:
            raise TypeError(f"Unsupported feature type: {type(feature)}")

        # Validation: check for NaN/Inf
        if not (np.isfinite(val[0]) and np.isfinite(val[1])):
            raise ValueError(f"Invalid non-finite feature value: {val}")

        self.buffer.append(val)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Check if buffer has accumulated full sequence length (e.g. 60 frames)."""
        return len(self.buffer) >= self.sequence_length

    def get_sequence(self, normalize: bool = True) -> torch.FloatTensor:
        """
        Extract buffered sequence, apply training z-score normalization,
        and convert to torch FloatTensor of shape (1, sequence_length, 2).

        Returns:
            torch.FloatTensor: Shape (1, sequence_length, 2)
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer is not ready yet. Current size: {len(self.buffer)}/{self.sequence_length}"
            )

        # Convert to numpy array of shape (60, 2)
        arr = np.array(self.buffer, dtype=np.float32)

        # Validate shape
        if arr.shape != (self.sequence_length, 2):
            raise ValueError(f"Unexpected buffer shape: {arr.shape}, expected ({self.sequence_length}, 2)")

        # Validate finite values
        if not np.all(np.isfinite(arr)):
            raise ValueError("Buffer contains NaN or Inf values.")

        # Z-score normalization: (feature - train_mean) / train_std
        if normalize:
            arr = (arr - self.mean) / self.std

        # Shape transformation: (60, 2) -> (1, 60, 2)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()

        # Final sanity checks on tensor
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Output tensor contains NaN or Inf.")

        return tensor

    def clear(self):
        """Clear all stored frames in buffer."""
        self.buffer.clear()
