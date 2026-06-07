"""Weighted ensemble: LSTM + XGBoost + Transformer (+ optional RL agent).

The :class:`EnsemblePredictor` accepts already‑constructed sub‑model
objects so that callers can swap implementations freely. Each sub‑model
must expose ``predict(features) -> dict`` with at least the keys
``direction`` (long/short), ``expected_return_pct``, and ``confidence``.
Additionally, any of them may provide ``iv_change_pct``.

Default weights when RL agent is absent (sum to 1.0):

    LSTM         0.30
    XGBoost      0.40
    Transformer  0.30

When an RL agent is present, weights are rescaled:

    LSTM         0.24
    XGBoost      0.32
    Transformer  0.24
    RL           0.20
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class SubModel(Protocol):
    """Structural interface a sub‑model must satisfy."""

    def predict(self, features: Any) -> Dict[str, Any]: ...


@dataclass
class EnsembleWeights:
    lstm: float = 0.30
    xgboost: float = 0.40
    transformer: float = 0.30
    rl: float = 0.0   # non-zero only when an RL agent is provided

    def normalised(self) -> Dict[str, float]:
        total = self.lstm + self.xgboost + self.transformer + self.rl
        if total <= 0:
            raise ValueError("Ensemble weights must be positive.")
        return {
            "lstm":        self.lstm        / total,
            "xgboost":     self.xgboost     / total,
            "transformer": self.transformer / total,
            "rl":          self.rl          / total,
        }


@dataclass
class EnsemblePredictor:
    lstm: SubModel
    xgboost: SubModel
    transformer: SubModel
    weights: EnsembleWeights = field(default_factory=EnsembleWeights)
    rl: Optional[SubModel] = None   # loaded from models/rl_ppo.zip when present

    def __post_init__(self) -> None:
        # Auto-load the RL agent if the model file exists and none was provided.
        if self.rl is None:
            try:
                from src.rl.rl_agent import RLAgent
                loaded = RLAgent.load_if_exists()
                if loaded is not None:
                    self.rl = loaded
                    # Allocate 20% to RL, shrink others proportionally.
                    self.weights = EnsembleWeights(
                        lstm=0.24, xgboost=0.32, transformer=0.24, rl=0.20
                    )
                    logger.info("Ensemble: RL agent loaded (weights lstm=0.24 xgb=0.32 tf=0.24 rl=0.20)")
            except Exception as exc:
                logger.debug("RL agent not loaded: %s", exc)

    def _direction_score(self, direction: str) -> float:
        if direction not in ("long", "short", "hold"):
            raise ValueError(f"Unexpected direction label: {direction!r}")
        if direction == "long":
            return 1.0
        if direction == "short":
            return -1.0
        return 0.0  # hold

    def predict(self, features: Any) -> Dict[str, Any]:
        """Run all sub‑models and aggregate.

        Returns a dict with keys ``direction``, ``expected_return_pct``,
        ``iv_change_pct``, and ``confidence`` (in [0, 1]).
        """
        sub_outs: Dict[str, Dict[str, Any]] = {}

        members = [
            ("lstm",        self.lstm),
            ("xgboost",     self.xgboost),
            ("transformer", self.transformer),
        ]
        if self.rl is not None:
            members.append(("rl", self.rl))

        for name, model in members:
            try:
                sub_outs[name] = model.predict(features)
            except Exception as exc:
                logger.warning("Sub-model %s failed: %s", name, exc)
                sub_outs[name] = {"direction": "long", "confidence": 0.5,
                                  "expected_return_pct": 0.0}

        w = self.weights.normalised()
        # Only sum weights for active members
        active_w = {n: w[n] for n in sub_outs}
        total_w  = sum(active_w.values())
        active_w = {n: v / total_w for n, v in active_w.items()}

        # Weighted directional vote in [-1, 1].
        # We center each sub-model's confidence at 0.5 = neutral so that
        # a model reporting 0.5 confidence contributes 0 to the directional
        # vote (instead of 0.5×direction, which overstates certainty).
        # Formula: (2×conf − 1) maps [0→−1, 0.5→0, 1→+1].
        directional = sum(
            active_w[name] * (2.0 * sub["confidence"] - 1.0) * self._direction_score(sub["direction"])
            for name, sub in sub_outs.items()
        )
        direction = "long" if directional >= 0 else "short"

        # Weighted expected return.
        expected_return_pct = float(
            sum(active_w[name] * float(sub.get("expected_return_pct", 0.0))
                for name, sub in sub_outs.items())
        )

        # IV change is only emitted by models that produce it; ignore otherwise.
        iv_terms = [
            (active_w[name], float(sub.get("iv_change_pct", np.nan)))
            for name, sub in sub_outs.items()
        ]
        iv_terms = [(wi, vi) for wi, vi in iv_terms if not np.isnan(vi)]
        if iv_terms:
            iv_w_sum = sum(wi for wi, _ in iv_terms)
            iv_change_pct = sum(wi * vi for wi, vi in iv_terms) / max(iv_w_sum, 1e-9)
        else:
            iv_change_pct = 0.0

        # Confidence: weighted average of individual model certainties,
        # scaled by directional alignment.
        base_conf = float(sum(active_w[name] * sub["confidence"] for name, sub in sub_outs.items()))
        max_possible = float(sum(active_w[name] * abs(2.0 * sub["confidence"] - 1.0)
                                 for name, sub in sub_outs.items()))
        agreement = min(1.0, abs(directional) / max(max_possible, 1e-6))
        confidence = float(min(1.0, base_conf * max(0.60, agreement)))

        result = {
            "direction": direction,
            "expected_return_pct": expected_return_pct,
            "iv_change_pct": float(iv_change_pct),
            "confidence": confidence,
            "components": sub_outs,
        }
        logger.debug("Ensemble prediction: %s", result)
        return result
