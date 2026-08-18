"""
tests/test_model.py

Test skrip untuk memverifikasi kompatibilitas checkpoint best_landmark_gru_ear_mar.pth
dan arsitektur LandmarkGRU sebelum aplikasi realtime dijalankan.
"""

import os
import sys
import json
import unittest
import torch

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import create_model, load_model, LandmarkGRU


class TestLandmarkGRU(unittest.TestCase):
    """
    Test suite untuk memverifikasi model LandmarkGRU dan checkpoint.
    """

    def setUp(self):
        self.config_path = os.path.join("models", "FINAL_landmark_gru_config.json")
        self.checkpoint_path = os.path.join("models", "best_landmark_gru_ear_mar.pth")

    def test_checkpoint_exists(self):
        self.assertTrue(os.path.exists(self.config_path), f"Config file missing: {self.config_path}")
        self.assertTrue(os.path.exists(self.checkpoint_path), f"Checkpoint file missing: {self.checkpoint_path}")

    def test_load_and_inference(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model, metadata = load_model(self.checkpoint_path, device="cpu", config_path=self.config_path)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, LandmarkGRU)

        seq_len = config.get("sequence_length", 60)
        input_dim = config.get("input_dim", 2)
        dummy_input = torch.randn(1, seq_len, input_dim)

        with torch.no_grad():
            logits = model(dummy_input)
            probabilities = torch.softmax(logits, dim=-1)

        self.assertEqual(logits.shape, (1, 2))
        self.assertFalse(torch.isnan(probabilities).any())
        self.assertFalse(torch.isinf(probabilities).any())
        self.assertTrue((probabilities >= 0.0).all() and (probabilities <= 1.0).all())
        self.assertTrue(torch.isclose(torch.sum(probabilities), torch.tensor(1.0), atol=1e-5))


def run_model_test():
    print("=" * 60)
    print("TESTING LANDMARK GRU MODEL & CHECKPOINT COMPATIBILITY")
    print("=" * 60)

    config_path = os.path.join("models", "FINAL_landmark_gru_config.json")
    checkpoint_path = os.path.join("models", "best_landmark_gru_ear_mar.pth")

    # 1. Load config
    print(f"[1/7] Loading configuration from: {config_path}")
    assert os.path.exists(config_path), f"Configuration file not found: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"      Model type: {config.get('model')}")
    print(f"      Features: {config.get('features')}")
    print(f"      Hidden dim: {config.get('hidden_dim')}, GRU layers: {config.get('gru_layers')}")

    # 2. Check checkpoint existence
    print(f"[2/7] Checking checkpoint file: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    file_size_kb = os.path.getsize(checkpoint_path) / 1024
    print(f"      Checkpoint size: {file_size_kb:.2f} KB")

    # 3 & 4. Load model & state dict
    print("[3/7] Instantiating LandmarkGRU and loading state_dict with strict=True...")
    try:
        model, metadata = load_model(checkpoint_path, device="cpu", config_path=config_path)
        print("      State dict loaded successfully without mismatch.")
        print(f"      Metadata from checkpoint:")
        print(f"        - Feature mean: {metadata.get('feature_mean')}")
        print(f"        - Feature std: {metadata.get('feature_std')}")
        print(f"        - Sequence length: {metadata.get('sequence_length')}")
    except Exception as e:
        print(f"[ERROR] Failed to load model state dict: {e}")
        raise

    # 5. Dummy input
    seq_len = config.get("sequence_length", 60)
    input_dim = config.get("input_dim", 2)
    dummy_input = torch.randn(1, seq_len, input_dim)
    print(f"[4/7] Generating dummy input tensor with shape: {tuple(dummy_input.shape)}")

    # 6. Run Inference
    print("[5/7] Executing forward pass...")
    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = torch.softmax(logits, dim=-1)

    # 7. Print results
    prob_np = probabilities[0].numpy()
    pred_class = int(torch.argmax(probabilities, dim=-1).item())
    class_names = {0: "NOT DROWSY", 1: "DROWSY"}

    print("[6/7] Inference Results:")
    print(f"      Input shape        : {tuple(dummy_input.shape)}")
    print(f"      Output logits shape: {tuple(logits.shape)}")
    print(f"      Probabilities      : [NOT DROWSY: {prob_np[0]:.4f}, DROWSY: {prob_np[1]:.4f}]")
    print(f"      Predicted Class    : {pred_class} ({class_names[pred_class]})")

    # 8. Assertions
    print("[7/7] Validating assertions...")
    assert logits.shape == (1, 2), f"Expected output shape (1, 2), got {logits.shape}"
    assert not torch.isnan(probabilities).any(), "Probabilities contain NaN values"
    assert not torch.isinf(probabilities).any(), "Probabilities contain Inf values"
    assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all(), "Probabilities out of [0, 1] range"
    assert torch.isclose(torch.sum(probabilities), torch.tensor(1.0), atol=1e-5), "Probabilities do not sum to 1.0"

    print("=" * 60)
    print("ALL TESTS PASSED! Model is fully compatible and ready for inference.")
    print("=" * 60)


if __name__ == "__main__":
    run_model_test()
