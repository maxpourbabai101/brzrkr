"""Weighted ensemble: LSTM + XGBoost + Transformer.

The :class:`EnsemblePredictor` accepts already‑constructed sub‑model
objects so that callers can swap implementations freely. Each sub‑model
must expose ``predict(features) -> dict`` with at least the keys
``direction`` (long/short), ``expected_return_pct``, and ``confidence``.
Additionally, any of them may provide ``iv_change_pct``.

Default weights (sum to 1.0):

    LSTM         0.30
    XGBoost      0.40
    Transformer  0.30
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

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

    def normalised(self) -> Dict[str, float]:
        total = self.lstm + self.xgboost + self.transformer
        if total <= 0:
            raise ValueError("Ensemble weights must be positive.")
        return {
            "lstm": self.lstm / total,
            "xgboost": self.xgboost / total,
            "transformer": self.transformer / total,
        }


@dataclass
class EnsemblePredictor:
    lstm: SubModel
    xgboost: SubModel
    transformer: SubModel
    weights: EnsembleWeights = field(default_factory=EnsembleWeights)

    def _direction_score(self, direction: str) -> float:
        if direction not in ("long", "short"):
            raise ValueError(f"Unexpected direction label: {direction!r}")
        return 1.0 if direction == "long" else -1.0

    def predict(self, features: Any) -> Dict[str, Any]:
        """Run all sub‑models and aggregate.

        Returns a dict with keys ``direction``, ``expected_return_pct``,
        ``iv_change_pct``, and ``confidence`` (in [0, 1]).
        """
        sub_outs: Dict[str, Dict[str, Any]] = {
            "lstm": self.lstm.predict(features),
            "xgboost": self.xgboost.predict(features),
            "transformer": self.transformer.predict(features),
        }
        w = self.weights.normalised()

        # Weighted directional vote in [-1, 1].
        directional = sum(
            w[name] * sub["confidence"] * self._direction_score(sub["direction"])
            for name, sub in sub_outs.items()
        )
        direction = "long" if directional >= 0 else "short"

        # Weighted expected return.
        expected_return_pct = float(
            sum(w[name] * float(sub.get("expected_return_pct", 0.0))
                for name, sub in sub_outs.items())
        )

        # IV change is only emitted by models that produce it; ignore otherwise.
        iv_terms = [
            (w[name], float(sub.get("iv_change_pct", np.nan)))
            for name, sub in sub_outs.items()
        ]
        iv_terms = [(wi, vi) for wi, vi in iv_terms if not np.isnan(vi)]
        if iv_terms:
            iv_w_sum = sum(wi for wi, _ in iv_terms)
            iv_change_pct = sum(wi * vi for wi, vi in iv_terms) / max(iv_w_sum, 1e-9)
        else:
            iv_change_pct = 0.0

        # Confidence: weighted average of individual model certainties,
        # scaled by directional alignment (how much do models agree on direction?).
        #
        # BUG FIX — old formula `abs(directional)` collapsed to ~0.10-0.22 whenever
        # even ONE model disagreed on direction (e.g. Transformer said SHORT while
        # LSTM + XGBoost said LONG). This made the threshold unreachable in practice.
        #
        # New formula:
        #   base_conf = weighted average of per-model confidence values
        #   agreement = |directional_vote| / base_conf  (0 = models split, 1 = all agree)
        #   confidence = base_conf * max(0.35, agreement)
        #
        # Effect: full agreement → confidence = base_conf (~0.55-0.80)
        #         mild disagreement → conference ≥ 0.35 * base_conf (~0.20)
        #         This is a meaningful signal, not a near-zero artefact.
        base_conf = float(sum(w[name] * sub["confidence"] for name, sub in sub_outs.items()))
        agreement = min(1.0, abs(directional) / max(base_conf, 1e-6))
        confidence = float(min(1.0, base_conf * max(0.35, agreement)))

        result = {
            "direction": direction,
            "expected_return_pct": expected_return_pct,
            "iv_change_pct": float(iv_change_pct),
            "confidence": confidence,
            "components": sub_outs,
        }
        logger.debug("Ensemble prediction: %s", result)
        return result
