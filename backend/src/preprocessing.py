"""
backend/src/preprocessing.py

Preprocessing and buffering for EAR + MAR sequence data:
- Z-score normalization using TRAINING mean and std (from checkpoint/config)
- Buffer maintaining sliding window of 60 frames
- Conversion to torch.FloatTensor with shape (1, 60, 2)
"""

import os
import json
from collections import deque
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from src.config import DEFAULT_FEATURE_MEAN, DEFAULT_FEATURE_STD
DEFAULT_SEQUENCE_LENGTH = 60


def load_training_statistics(
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    mean = np.array(DEFAULT_FEATURE_MEAN, dtype=np.float32)
    std = np.array(DEFAULT_FEATURE_STD, dtype=np.float32)
    seq_len = 60

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

        self.std = np.where(self.std < 1e-7, 1e-7, self.std)
        self.buffer = deque(maxlen=self.sequence_length)

    def append(self, feature: Union[List[float], Tuple[float, float], np.ndarray]):
        if isinstance(feature, (list, tuple)):
            if len(feature) < 2:
                raise ValueError(f"Feature must contain at least 2 elements [EAR, MAR], got: {feature}")
            val = [float(feature[0]), float(feature[1])]
        elif isinstance(feature, np.ndarray):
            val = [float(feature[0]), float(feature[1])]
        else:
            raise TypeError(f"Unsupported feature type: {type(feature)}")

        if not (np.isfinite(val[0]) and np.isfinite(val[1])):
            raise ValueError(f"Invalid non-finite feature value: {val}")

        self.buffer.append(val)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.sequence_length

    def get_sequence(self, normalize: bool = True) -> torch.FloatTensor:
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer is not ready yet. Current size: {len(self.buffer)}/{self.sequence_length}"
            )

        arr = np.array(self.buffer, dtype=np.float32)

        if arr.shape != (self.sequence_length, 2):
            raise ValueError(f"Unexpected buffer shape: {arr.shape}, expected ({self.sequence_length}, 2)")

        if not np.all(np.isfinite(arr)):
            raise ValueError("Buffer contains NaN or Inf values.")

        if normalize:
            arr = (arr - self.mean) / self.std

        tensor = torch.from_numpy(arr).unsqueeze(0).float()

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError("Output tensor contains NaN or Inf.")

        return tensor

    def clear(self):
        self.buffer.clear()
