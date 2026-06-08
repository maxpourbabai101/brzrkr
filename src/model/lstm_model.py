"""LSTM model for directional price prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

SEQ_LEN = 256


@dataclass
class LSTMConfig:
    input_dim: int = 8          # OHLCV + engineered features
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.2
    seq_len: int = SEQ_LEN
    bidirectional: bool = False


class LSTMPredictor(nn.Module):
    """LSTM for sequence classification + regression."""

    def __init__(self, config: LSTMConfig | None = None) -> None:
        super().__init__()
        self.config = config or LSTMConfig()

        self.input_proj = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.n_layers,
            dropout=self.config.dropout if self.config.n_layers > 1 else 0.0,
            bidirectional=self.config.bidirectional,
            batch_first=True,
        )
        lstm_out_dim = self.config.hidden_dim * (2 if self.config.bidirectional else 1)

        # Two heads: direction (2 classes) + magnitude
        self.direction_head = nn.Linear(lstm_out_dim, 2)
        self.magnitude_head = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x shape: (batch, seq_len, input_dim). Returns (direction_logits, magnitude_pct)."""
        if x.dim() != 3:
            raise ValueError(f"expected 3D tensor (B,T,F); got shape {tuple(x.shape)}")

        h = self.input_proj(x)
        lstm_out, _ = self.lstm(h)
        pooled = lstm_out[:, -1, :]  # Last time step
        return self.direction_head(pooled), self.magnitude_head(pooled).squeeze(-1)

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> Dict[str, float | str]:
        """Run inference on numpy window of shape (seq_len, F)."""
        if window.ndim != 2 or window.shape[0] != self.config.seq_len:
            raise ValueError(
                f"window must have shape ({self.config.seq_len}, F); got {window.shape}"
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