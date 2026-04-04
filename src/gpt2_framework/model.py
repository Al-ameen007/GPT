import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, drop_rate, n_heads, qkv_bias):
        super().__init__()

        assert d_out % n_heads == 0, (
            f"d_out ({d_out}) must be divisible by n_heads ({n_heads})"
        )

        self.d_out = d_out
        self.d_in = d_in
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.W_query = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_key = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_value = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(p=drop_rate)
        self.out_proj = nn.Linear(d_out, d_out)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values

        context_vector = context_vector.transpose(1, 2)
        context_vector = context_vector.contiguous().view(
            batch_size, num_tokens, self.d_out
        )

        context_vector = self.out_proj(context_vector)

        return context_vector


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        norm_x = self.scale * norm_x + self.shift

        return norm_x


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_len=cfg["context_length"],
            drop_rate=cfg["drop_rate"],
            n_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_layer = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        x = self.trf_blocks(x)
        x = self.final_layer(x)
        logits = self.out_head(x)

        return logits


class GPTClassifier(nn.Module):
    def __init__(self, cfg, num_classes=2):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_layer = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], num_classes, bias=True
        )  # Changed to bias=True

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        x = self.trf_blocks(x)
        x = self.final_layer(x)
        logits = self.out_head(x)

        return logits
