"""Tests for checkpoint save/load."""

import torch

from gpt2_framework.checkpoint import save_model
from gpt2_framework.config import GPT_CONFIG_SMALL
from gpt2_framework.model import GPTModel


def test_save_and_load_model(tmp_path):
    """Model should survive a save/load round-trip."""
    cfg = GPT_CONFIG_SMALL.copy()
    cfg["context_length"] = 64
    cfg["n_layers"] = 2
    cfg["qkv_bias"] = False

    model = GPTModel(cfg)
    path = tmp_path / "test_model.pt"
    save_model(model, path, cfg)

    checkpoint = torch.load(path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "config" in checkpoint

    loaded = GPTModel(checkpoint["config"])
    loaded.load_state_dict(checkpoint["model_state_dict"])
