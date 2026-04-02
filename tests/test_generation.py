"""Tests for text generation utilities."""

import tiktoken
import torch

from gpt2_framework.generation import text_to_token, token_ids_to_text


def test_round_trip_tokenization():
    """Encoding then decoding should return the original text."""
    tokenizer = tiktoken.get_encoding("gpt2")
    text = "Hello, world!"
    token_ids = text_to_token(text, tokenizer)
    assert isinstance(token_ids, torch.Tensor)
    decoded = token_ids_to_text(token_ids, tokenizer)
    assert decoded.strip() == text
