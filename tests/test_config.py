"""Tests for model configurations."""

import pytest

from gpt2_framework.config import (
    GPT_CONFIG_LARGE,
    GPT_CONFIG_MEDIUM,
    GPT_CONFIG_SMALL,
    GPT_CONFIG_XLARGE,
)

REQUIRED_KEYS = {
    "vocab_size",
    "context_length",
    "emb_dim",
    "n_heads",
    "n_layers",
    "drop_rate",
    "qkv_bias",
}
ALL_CONFIGS = [GPT_CONFIG_SMALL, GPT_CONFIG_MEDIUM, GPT_CONFIG_LARGE, GPT_CONFIG_XLARGE]


@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_config_has_required_keys(config):
    """Every config must contain the full set of required keys."""
    assert REQUIRED_KEYS.issubset(config.keys())


@pytest.mark.parametrize("config", ALL_CONFIGS)
def test_emb_dim_divisible_by_heads(config):
    """emb_dim must be evenly divisible by n_heads for multi-head attention."""
    assert config["emb_dim"] % config["n_heads"] == 0
