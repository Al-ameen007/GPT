"""Tests for GPT model architecture."""

import pytest
import torch

from gpt2_framework.config import GPT_CONFIG_SMALL
from gpt2_framework.model import GPTClassifier, GPTModel


@pytest.fixture
def small_config():
    """A minimal config for fast test runs."""
    cfg = GPT_CONFIG_SMALL.copy()
    cfg["context_length"] = 64  # Short context for speed
    cfg["n_layers"] = 2  # Fewer layers for speed
    cfg["qkv_bias"] = False
    return cfg


def test_gpt_model_output_shape(small_config):
    """GPTModel output should be (batch, seq_len, vocab_size)."""
    model = GPTModel(small_config)
    model.eval()
    x = torch.randint(0, small_config["vocab_size"], (2, 32))  # batch=2, seq=32
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 32, small_config["vocab_size"])


def test_gpt_classifier_output_shape(small_config):
    """GPTClassifier output should be (batch, seq_len, num_classes)."""
    num_classes = 3
    model = GPTClassifier(small_config, num_classes=num_classes)
    model.eval()
    x = torch.randint(0, small_config["vocab_size"], (2, 32))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 32, num_classes)
