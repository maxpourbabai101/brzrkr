"""Tests for src.model.ensemble.EnsemblePredictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from src.model.ensemble import EnsemblePredictor, EnsembleWeights


@dataclass
class _Stub:
    direction: str
    expected_return_pct: float
    confidence: float
    iv_change_pct: float = 0.0

    def predict(self, _features: Any) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "expected_return_pct": self.expected_return_pct,
            "iv_change_pct": self.iv_change_pct,
            "confidence": self.confidence,
        }


def test_ensemble_majority_long():
    ens = EnsemblePredictor(
        lstm=_Stub("long", 0.01, 0.8),
        xgboost=_Stub("long", 0.012, 0.9),
        transformer=_Stub("short", 0.005, 0.6),
    )
    out = ens.predict(features=None)
    assert out["direction"] == "long"
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["expected_return_pct"] > 0


def test_ensemble_majority_short():
    ens = EnsemblePredictor(
        lstm=_Stub("short", -0.008, 0.85),
        xgboost=_Stub("short", -0.011, 0.92),
        transformer=_Stub("long", 0.003, 0.55),
    )
    out = ens.predict(features=None)
    assert out["direction"] == "short"
    assert out["confidence"] > 0


def test_weights_must_be_positive():
    with pytest.raises(ValueError):
        EnsembleWeights(0, 0, 0).normalised()


def test_iv_change_weighted_average():
    ens = EnsemblePredictor(
        lstm=_Stub("long", 0.01, 0.7, iv_change_pct=0.02),
        xgboost=_Stub("long", 0.01, 0.7, iv_change_pct=0.04),
        transformer=_Stub("long", 0.01, 0.7, iv_change_pct=0.06),
    )
    out = ens.predict(features=None)
    # Weighted IV = 0.3*0.02 + 0.4*0.04 + 0.3*0.06 = 0.04
    assert out["iv_change_pct"] == pytest.approx(0.04, abs=1e-6)
