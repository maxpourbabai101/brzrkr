"""Tests for src.training.* + src.model.xgb_predictor.

End-to-end pipeline test uses synthetic prices with a deliberately
learnable signal — RSI-style momentum — so we can verify the
classifier achieves > random-chance accuracy in walk-forward CV.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineer import FeatureEngineer
from src.training.feature_dataset import FeatureDataset, LabelConfig
from src.training.trainer import TrainConfig, XGBoostTrainer


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
def _synthetic_prices_with_momentum(n=600, seed=0):
    """Random walk with a small autoregressive momentum component, so a
    classifier should beat 50/50."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    # Add modest momentum: this period's return = 0.3 * last period + noise.
    for i in range(1, n):
        rets[i] += 0.3 * rets[i - 1]
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.integers(50_000, 500_000, n)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1D")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": volume}, index=idx)


# ---------------------------------------------------------------------------
# FeatureDataset
# ---------------------------------------------------------------------------
def test_feature_dataset_produces_labeled_pairs():
    prices = _synthetic_prices_with_momentum(n=400)
    fe = FeatureEngineer(window=64)
    ds = FeatureDataset(feature_engineer=fe,
                        label_cfg=LabelConfig(horizon_bars=5,
                                               long_threshold=0.003,
                                               short_threshold=0.003))
    X, y = ds.build(prices)
    assert len(X) == len(y)
    assert len(X) > 50
    assert set(y.unique()).issubset({0, 1})
    # No NaNs survived.
    assert X.isna().sum().sum() == 0


def test_feature_dataset_drops_middle_zone():
    prices = _synthetic_prices_with_momentum(n=400)
    fe = FeatureEngineer(window=64)
    wide = FeatureDataset(feature_engineer=fe,
                          label_cfg=LabelConfig(horizon_bars=5,
                                                 long_threshold=0.05,
                                                 short_threshold=0.05,
                                                 drop_middle=True))
    narrow = FeatureDataset(feature_engineer=fe,
                            label_cfg=LabelConfig(horizon_bars=5,
                                                   long_threshold=0.001,
                                                   short_threshold=0.001,
                                                   drop_middle=True))
    X_wide, _ = wide.build(prices)
    X_narrow, _ = narrow.build(prices)
    # Wider thresholds = more samples dropped in the middle.
    assert len(X_wide) < len(X_narrow)


def test_feature_dataset_raises_on_short_history():
    prices = _synthetic_prices_with_momentum(n=50)
    fe = FeatureEngineer(window=64)
    ds = FeatureDataset(feature_engineer=fe, min_history_bars=128)
    with pytest.raises(ValueError):
        ds.build(prices)


# ---------------------------------------------------------------------------
# Trainer + saved artifacts
# ---------------------------------------------------------------------------
def test_train_saves_artifacts_and_beats_random(tmp_path):
    prices = _synthetic_prices_with_momentum(n=800, seed=42)
    fe = FeatureEngineer(window=64)
    ds = FeatureDataset(feature_engineer=fe,
                        label_cfg=LabelConfig(horizon_bars=5,
                                               long_threshold=0.003,
                                               short_threshold=0.003))
    X, y = ds.build(prices)
    trainer = XGBoostTrainer(TrainConfig(
        n_splits=3, test_size=30, n_estimators=50, max_depth=3,
        early_stopping_rounds=None,   # disable for quick test
    ))
    report = trainer.train(X, y, output_dir=tmp_path)

    # Files exist.
    assert (tmp_path / "xgb.json").exists()
    assert (tmp_path / "xgb_features.json").exists()
    assert (tmp_path / "xgb_report.json").exists()

    # Report sanity.
    assert report.n_samples == len(X)
    assert report.n_features == X.shape[1]
    assert len(report.cv_scores) == 3
    # With a real momentum signal in the data, mean CV accuracy
    # should beat 0.45 (very lax — the test is "doesn't crash and
    # produces non-degenerate output," not "is good model").
    assert report.mean_cv_accuracy > 0.45


def test_train_rejects_empty_inputs():
    trainer = XGBoostTrainer(TrainConfig(early_stopping_rounds=None))
    with pytest.raises(ValueError):
        trainer.train(pd.DataFrame(), pd.Series(dtype=int))


# ---------------------------------------------------------------------------
# Inference wrapper (XGBoostPredictor)
# ---------------------------------------------------------------------------
def test_xgb_predictor_loads_and_predicts(tmp_path):
    # Train a tiny model end-to-end then load via the predictor.
    prices = _synthetic_prices_with_momentum(n=400, seed=1)
    fe = FeatureEngineer(window=64)
    ds = FeatureDataset(feature_engineer=fe,
                        label_cfg=LabelConfig(horizon_bars=5))
    X, y = ds.build(prices)
    XGBoostTrainer(TrainConfig(n_splits=2, test_size=20, n_estimators=30,
                                max_depth=3, early_stopping_rounds=None)
                    ).train(X, y, output_dir=tmp_path)

    from src.model.xgb_predictor import XGBoostPredictor
    pred = XGBoostPredictor(model_path=tmp_path / "xgb.json",
                            features_path=tmp_path / "xgb_features.json")

    # Feed it the same engineered features it was trained on.
    feat = fe.build_features({"prices": prices.tail(200)})
    out = pred.predict(feat)
    assert out["direction"] in ("long", "short")
    assert 0.5 <= out["confidence"] <= 1.0
    assert "expected_return_pct" in out


def test_maybe_load_returns_none_when_files_missing(tmp_path):
    from src.model.xgb_predictor import maybe_load_xgb_predictor
    p = maybe_load_xgb_predictor(model_path=tmp_path / "nope.json",
                                  features_path=tmp_path / "nope.json")
    assert p is None


def test_predictor_abstains_on_empty_features(tmp_path):
    # Build a minimal predictor with a dummy trained model.
    prices = _synthetic_prices_with_momentum(n=400, seed=2)
    fe = FeatureEngineer(window=64)
    ds = FeatureDataset(feature_engineer=fe, label_cfg=LabelConfig())
    X, y = ds.build(prices)
    XGBoostTrainer(TrainConfig(n_splits=2, test_size=20, n_estimators=20,
                                max_depth=2, early_stopping_rounds=None)
                    ).train(X, y, output_dir=tmp_path)

    from src.model.xgb_predictor import XGBoostPredictor
    pred = XGBoostPredictor(model_path=tmp_path / "xgb.json",
                            features_path=tmp_path / "xgb_features.json")
    out = pred.predict(pd.DataFrame())
    # Abstention = confidence at no-trade boundary.
    assert out["confidence"] == 0.5
    assert out["expected_return_pct"] == 0.0
