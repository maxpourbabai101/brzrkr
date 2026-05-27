"""Transformer (TFT‑style) backbone for directional price prediction.

A 3‑layer encoder with sinusoidal positional encoding and multi‑head
self‑attention. Consumes a 256‑step window of price/volume features
and emits a direction (long/short) plus an expected percentage move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 256


@dataclass
class TransformerConfig:
    input_dim: int = 8        # e.g., OHLCV + 3 engineered features
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    seq_len: int = SEQ_LEN


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerPredictor(nn.Module):
    """3‑layer Transformer that predicts direction + magnitude."""

    def __init__(self, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransformerConfig()

        self.input_proj = nn.Linear(self.config.input_dim, self.config.d_model)
        self.pos_enc = _PositionalEncoding(self.config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.n_layers
        )

        # Two heads: a 2‑class direction logit and a continuous magnitude.
        self.direction_head = nn.Linear(self.config.d_model, 2)
        self.magnitude_head = nn.Linear(self.config.d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`x` shape: (batch, seq_len, input_dim).

        Returns ``(direction_logits, magnitude_pct)``.
        """
        if x.dim() != 3:
            raise ValueError(f"expected 3D tensor (B,T,F); got shape {tuple(x.shape)}")
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        pooled = h[:, -1, :]  # Use last‑step representation
        return self.direction_head(pooled), self.magnitude_head(pooled).squeeze(-1)

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> Dict[str, float | str]:
        """Run a single inference on a numpy window of shape (256, F).

        Returns ``{"direction": "long"|"short", "magnitude_pct": float,
        "confidence": float}``.
        """
        if window.ndim != 2 or window.shape[0] != self.config.seq_len:
            raise ValueError(
                f"window must have shape ({self.config.seq_len}, F); "
                f"got {window.shape}"
            )
        self.eval()
        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
        logits, mag = self.forward(x)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        direction_idx = int(torch.argmax(probs).item())
        return {
            "direction": "long" if direction_idx == 1 else "short",
            "magnitude_pct": float(mag.item()),
            "confidence": float(probs[direction_idx].item()),
        }
