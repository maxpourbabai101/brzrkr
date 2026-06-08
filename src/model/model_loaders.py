"""Model loaders for trained PyTorch models (LSTM, Transformer)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.model import LSTMPredictor, LSTMConfig, TransformerPredictor, TransformerConfig

logger = logging.getLogger(__name__)

# Default paths for trained models
DEFAULT_LSTM_PATH = Path("models/lstm_model.pt")
DEFAULT_LSTM_CONFIG_PATH = Path("models/lstm_config.json")
DEFAULT_TRANSFORMER_PATH = Path("models/transformer_model.pt")
DEFAULT_TRANSFORMER_CONFIG_PATH = Path("models/transformer_config.json")


class _LSTMInferenceWrapper:
    """Wrapper to match the ensemble sub-model interface."""

    label = "lstm"

    def __init__(self, model: LSTMPredictor):
        self._model = model
        self._model.eval()

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        if features is None or features.empty or len(features) < self._model.config.seq_len:
            return _abstain()

        # Extract numeric feature columns matching the training input_dim
        # For now, use OHLCV + first 3 engineered features
        available_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        needed = self._model.config.input_dim

        # Priority columns for LSTM input
        priority = ["close", "open", "high", "low", "volume",
                    "ret_1", "ret_5", "ret_20", "sma_spread", "rsi", "vol_z",
                    "realized_vol", "sma_20", "sma_50"]
        selected = [c for c in priority if c in available_cols][:needed]

        if len(selected) < needed:
            logger.warning(f"LSTM: only {len(selected)}/{needed} input features available")
            # Pad with zeros
            while len(selected) < needed:
                selected.append(selected[-1] if selected else "close")

        window_df = features[selected].tail(self._model.config.seq_len)
        if len(window_df) < self._model.config.seq_len:
            return _abstain()

        # Normalize (simple min-max per feature to avoid scale issues)
        window = window_df.values.astype(np.float32)
        window = (window - window.mean(axis=0)) / (window.std(axis=0) + 1e-8)

        try:
            with torch.no_grad():
                x = torch.from_numpy(window).unsqueeze(0)
                logits, mag = self._model.forward(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                direction_idx = int(torch.argmax(probs).item())
                # Handle NaN from untrained magnitude head
                mag_val = float(mag.item())
                if not np.isfinite(mag_val):
                    mag_val = 0.0
                conf_val = float(probs[direction_idx].item())
                if not np.isfinite(conf_val):
                    conf_val = 0.5
                return {
                    "direction": "long" if direction_idx == 1 else "short",
                    "expected_return_pct": mag_val * 100,  # scale to percent
                    "iv_change_pct": 0.0,
                    "confidence": conf_val,
                }
        except Exception as exc:
            logger.warning(f"LSTM inference failed: {exc} — abstaining")
            return _abstain()


class _TransformerInferenceWrapper:
    """Wrapper to match the ensemble sub-model interface."""

    label = "transformer"

    def __init__(self, model: TransformerPredictor):
        self._model = model
        self._model.eval()

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        if features is None or features.empty or len(features) < self._model.config.seq_len:
            return _abstain()

        available_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        needed = self._model.config.input_dim

        priority = ["close", "open", "high", "low", "volume",
                    "ret_1", "ret_5", "ret_20", "sma_spread", "rsi", "vol_z",
                    "realized_vol", "sma_20", "sma_50"]
        selected = [c for c in priority if c in available_cols][:needed]

        if len(selected) < needed:
            while len(selected) < needed:
                selected.append(selected[-1] if selected else "close")

        window_df = features[selected].tail(self._model.config.seq_len)
        if len(window_df) < self._model.config.seq_len:
            return _abstain()

        window = window_df.values.astype(np.float32)
        window = (window - window.mean(axis=0)) / (window.std(axis=0) + 1e-8)

        try:
            with torch.no_grad():
                x = torch.from_numpy(window).unsqueeze(0)
                logits, mag = self._model.forward(x)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                direction_idx = int(torch.argmax(probs).item())
                # Handle NaN from untrained magnitude head
                mag_val = float(mag.item())
                if not np.isfinite(mag_val):
                    mag_val = 0.0
                conf_val = float(probs[direction_idx].item())
                if not np.isfinite(conf_val):
                    conf_val = 0.5
                return {
                    "direction": "long" if direction_idx == 1 else "short",
                    "expected_return_pct": mag_val * 100,
                    "iv_change_pct": 0.0,
                    "confidence": conf_val,
                }
        except Exception as exc:
            logger.warning(f"Transformer inference failed: {exc} — abstaining")
            return _abstain()


def _abstain() -> Dict[str, Any]:
    return {
        "direction": "long",
        "expected_return_pct": 0.0,
        "iv_change_pct": 0.0,
        "confidence": 0.5,
    }


def maybe_load_lstm(
    model_path: Path = DEFAULT_LSTM_PATH,
    config_path: Path = DEFAULT_LSTM_CONFIG_PATH,
) -> Optional[_LSTMInferenceWrapper]:
    """Load trained LSTM if checkpoint exists."""
    if not (model_path.exists() and config_path.exists()):
        return None
    try:
        import json
        config_data = json.loads(config_path.read_text())
        config = LSTMConfig(**config_data)
        model = LSTMPredictor(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        logger.info(f"Loaded TRAINED LSTM from {model_path}")
        return _LSTMInferenceWrapper(model)
    except Exception as exc:
        logger.warning(f"Could not load LSTM from {model_path}: {exc}")
        return None


def maybe_load_transformer(
    model_path: Path = DEFAULT_TRANSFORMER_PATH,
    config_path: Path = DEFAULT_TRANSFORMER_CONFIG_PATH,
) -> Optional[_TransformerInferenceWrapper]:
    """Load trained Transformer if checkpoint exists."""
    if not (model_path.exists() and config_path.exists()):
        return None
    try:
        import json
        config_data = json.loads(config_path.read_text())
        config = TransformerConfig(**config_data)
        model = TransformerPredictor(config)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        logger.info(f"Loaded TRAINED Transformer from {model_path}")
        return _TransformerInferenceWrapper(model)
    except Exception as exc:
        logger.warning(f"Could not load Transformer from {model_path}: {exc}")
        return None