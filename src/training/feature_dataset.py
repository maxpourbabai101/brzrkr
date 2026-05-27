"""FeatureDataset — convert historical OHLCV into (X, y) for training.

For each timestamp t in [min_history, N - horizon):

    X[t] = FeatureEngineer.build_features({"prices": prices[:t+1]}).iloc[-1]
    y[t] = 1  if forward_return(t, t+horizon) >  long_threshold
           0  if forward_return(t, t+horizon) < -short_threshold
           dropped otherwise (no-trade zone)

This is a simplified triple-barrier label (López de Prado, Advances in
Financial Machine Learning, Ch. 3). Samples in the "no-trade zone"
are dropped so the classifier learns only on clear directional moves.

Only the price-based features are used here. Sentiment / insider /
congress / wiki context is point-in-time and cannot be backfilled
cleanly without time-aligned historical archives, which the free
data scrapers don't provide. Add those in a v2 once you have a
historical news/sentiment archive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LabelConfig:
    horizon_bars: int = 5            # look this many bars ahead
    long_threshold: float = 0.005    # +0.5% over horizon => long label
    short_threshold: float = 0.005   # -0.5% over horizon => short label
    drop_middle: bool = True         # drop samples in the no-trade zone


@dataclass
class FeatureDataset:
    feature_engineer: object         # duck-typed: must have .build_features(bundle)
    label_cfg: LabelConfig = field(default_factory=LabelConfig)
    min_history_bars: int = 128      # need at least this many bars before first sample

    def build(self, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Return (X, y) ready for XGBoost.

        ``prices`` must be a datetime-indexed DataFrame with columns
        ``open, high, low, close, volume`` (case-insensitive).
        """
        prices = prices.copy()
        prices.columns = [c.lower() for c in prices.columns]
        if "close" not in prices.columns:
            raise ValueError("prices DataFrame missing 'close' column")

        N = len(prices)
        horizon = self.label_cfg.horizon_bars
        if N < self.min_history_bars + horizon + 1:
            raise ValueError(
                f"Need at least {self.min_history_bars + horizon + 1} bars; got {N}"
            )

        feature_rows: List[pd.Series] = []
        labels: List[int] = []
        kept_indices: List[pd.Timestamp] = []

        for t in range(self.min_history_bars, N - horizon):
            window = prices.iloc[:t + 1]
            bundle = {"prices": window}
            try:
                feats = self.feature_engineer.build_features(bundle)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skip t=%d: feature build failed (%s)", t, exc)
                continue
            if feats.empty:
                continue

            x = feats.iloc[-1]
            entry = float(prices["close"].iloc[t])
            exit_ = float(prices["close"].iloc[t + horizon])
            fwd_ret = (exit_ / entry) - 1.0

            if fwd_ret > self.label_cfg.long_threshold:
                y = 1
            elif fwd_ret < -self.label_cfg.short_threshold:
                y = 0
            else:
                if self.label_cfg.drop_middle:
                    continue
                y = 1 if fwd_ret >= 0 else 0

            feature_rows.append(x)
            labels.append(y)
            kept_indices.append(prices.index[t])

        if not feature_rows:
            raise RuntimeError(
                "No labeled samples produced. Try widening thresholds or "
                "extending the history."
            )

        X = pd.DataFrame(feature_rows, index=kept_indices)
        # Keep numeric columns only, drop columns that are all-NaN.
        X = X.select_dtypes(include=[np.number])
        X = X.dropna(axis=1, how="all")
        # Forward-fill remaining NaNs from rolling indicators; drop residual rows.
        X = X.ffill().dropna()
        y = pd.Series(labels, index=kept_indices, name="label").loc[X.index]
        logger.info("FeatureDataset: %d samples, %d features, "
                    "long ratio %.2f%%",
                    len(X), X.shape[1], (y.mean() * 100))
        return X, y
