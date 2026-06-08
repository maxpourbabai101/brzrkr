"""
DynamicWeightTracker — adjusts ensemble sub-model weights based on
recent directional accuracy.

Instead of a fixed lstm:0.30 / xgboost:0.40 / transformer:0.30,
the weights shift toward whichever sub-model has been most correct
in the last N predictions.

Storage: data/model_accuracy.json (persisted across restarts)
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

ACCURACY_PATH = Path("data/model_accuracy.json")
HISTORY_LEN   = 60    # track last 60 predictions per model
MIN_HISTORY   = 10    # need at least 10 samples before adjusting weights
ALPHA         = 0.60  # blend factor: 0 = always default, 1 = pure accuracy


class DynamicWeightTracker:
    """Tracks per-model accuracy and returns adjusted ensemble weights."""

    DEFAULTS: Dict[str, float] = {"lstm": 0.30, "xgboost": 0.40, "transformer": 0.30}

    def __init__(self) -> None:
        self._history: Dict[str, deque] = {
            k: deque(maxlen=HISTORY_LEN) for k in self.DEFAULTS
        }
        self._load()

    def _load(self) -> None:
        if not ACCURACY_PATH.exists():
            return
        try:
            data = json.loads(ACCURACY_PATH.read_text())
            for model, hits in data.items():
                if model in self._history:
                    for h in hits[-HISTORY_LEN:]:
                        self._history[model].append(h)
        except Exception:
            pass

    def _save(self) -> None:
        try:
            ACCURACY_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {k: list(v) for k, v in self._history.items()}
            ACCURACY_PATH.write_text(json.dumps(data))
        except Exception:
            pass

    def record(self, model_name: str, predicted_direction: str,
               actual_direction: str) -> None:
        """Call this after a trade closes to update model accuracy."""
        if model_name not in self._history:
            return
        correct = 1.0 if predicted_direction == actual_direction else 0.0
        self._history[model_name].append(correct)
        self._save()

    def get_weights(self) -> Dict[str, float]:
        """Return normalised weights for the ensemble."""
        accuracy: Dict[str, float] = {}
        for model, history in self._history.items():
            if len(history) >= MIN_HISTORY:
                acc = float(np.mean(list(history)))
                # edge above random: max(acc - 0.50, 0.01)
                accuracy[model] = max(acc - 0.50, 0.01)
            else:
                accuracy[model] = self.DEFAULTS[model]

        # Blend: ALPHA × accuracy-based + (1-ALPHA) × default
        blended: Dict[str, float] = {}
        total_acc = sum(accuracy.values())
        for model in self.DEFAULTS:
            acc_w = accuracy[model] / max(total_acc, 1e-9)
            blended[model] = ALPHA * acc_w + (1 - ALPHA) * self.DEFAULTS[model]

        total = sum(blended.values())
        weights = {k: v / total for k, v in blended.items()}

        logger.debug("Dynamic weights: %s", {k: f"{v:.3f}" for k, v in weights.items()})
        return weights


# Singleton
_tracker: DynamicWeightTracker | None = None


def get_weight_tracker() -> DynamicWeightTracker:
    global _tracker
    if _tracker is None:
        _tracker = DynamicWeightTracker()
    return _tracker
