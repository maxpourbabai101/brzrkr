"""
FeatureNormaliser — rolling cross-sectional Z-score normalisation.

Root cause of XGBoost's 52% accuracy: raw feature values (RSI, MACD,
price returns) are fed unnormalised across 22 stocks with completely
different price scales. NVDA at $900 and SPY at $500 look "different"
to a tree model even when both are in uptrends.

Fix: normalise every feature to its own 252-day rolling mean/std so
the model sees relative extremes, not absolute levels.

Usage (in FeatureEngineer or signal pipeline):
    from src.features.normalizer import FeatureNormaliser
    normed = FeatureNormaliser().transform(features_df)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features that should NOT be normalised (already in meaningful units
# or boolean flags that normalisation would destroy)
SKIP_COLS = {
    "close", "open", "high", "low", "volume",   # raw OHLCV — used for calculations, not features
    "direction",                                  # categorical
}

WINDOW = 252   # 1-year rolling window for normalisation statistics


class FeatureNormaliser:
    """Rolling Z-score normaliser for tabular price/indicator features."""

    def transform(self, df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
        """
        Apply rolling Z-score normalisation to all numeric columns except
        those in SKIP_COLS.

        Returns a new DataFrame with the same index and columns; NaN rows
        introduced by the rolling window are forward-filled from the first
        valid row.
        """
        if df is None or df.empty:
            return df

        out = df.copy()
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in SKIP_COLS
        ]

        for col in numeric_cols:
            series = df[col].astype(float)
            roll_mean = series.rolling(window, min_periods=20).mean()
            roll_std  = series.rolling(window, min_periods=20).std().replace(0, np.nan)

            z = (series - roll_mean) / roll_std

            # Winsorise at ±4σ to prevent outlier dominance
            z = z.clip(-4.0, 4.0)

            # Forward-fill the first window of NaN values
            z = z.ffill().fillna(0.0)

            out[col] = z

        logger.debug("FeatureNormaliser: normalised %d/%d columns",
                     len(numeric_cols), len(df.columns))
        return out

    def transform_last_row(self, df: pd.DataFrame, window: int = WINDOW) -> pd.Series:
        """Normalise and return only the last row — used at inference time."""
        normed = self.transform(df, window=window)
        return normed.iloc[-1]
