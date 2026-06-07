"""DrawdownPredictor — ML model that predicts near-term market drawdowns.

Trains an XGBoost binary classifier on historical price features.
Label: did the price drop more than ``dd_threshold`` in the next
``horizon`` bars?

The predictor exposes two public methods:

``predict_proba(features_df) -> float``
    Probability of a drawdown in the next ``horizon`` bars.
    Returns 0.5 if no model is loaded (neutral).

``size_multiplier(prob) -> float``
    Smooth position-size multiplier [``min_scale`` … 1.0].
    Passes full size through below a low-risk threshold, ramps
    down linearly to ``min_scale`` as probability approaches 1.

Auto-loading
------------
``DrawdownPredictor.load_if_exists()`` returns a loaded instance when
``models/drawdown_model.json`` exists, else None. The agent calls
this once at startup and caches the result.

Training
--------
Run ``python train_drawdown.py`` (CLI wrapper in the project root).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_PATH    = Path("models/drawdown_model.json")
_FEATURES_PATH = Path("models/drawdown_features.json")

# Features used for both training and inference.
FEATURE_COLS = [
    "ret_1", "ret_5", "ret_20",
    "sma_spread", "realized_vol",
    "rsi", "vol_z",
]

# Sizing thresholds
_LOW_RISK_PROB  = 0.30   # below this → full size (1.0×)
_HIGH_RISK_PROB = 0.70   # above this → minimum size (min_scale×)


class DrawdownPredictor:
    """XGBoost drawdown probability model."""

    def __init__(
        self,
        dd_threshold: float = 0.015,   # 1.5% drop counts as a drawdown
        horizon: int = 5,              # look-ahead bars
        min_scale: float = 0.35,       # minimum position-size multiplier
    ) -> None:
        self.dd_threshold = dd_threshold
        self.horizon      = horizon
        self.min_scale    = min_scale
        self._model       = None
        self._feature_cols: List[str] = FEATURE_COLS

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df_list: List[pd.DataFrame],
        *,
        n_splits: int = 5,
        n_estimators: int = 300,
        max_depth: int = 4,
        save_path: Optional[Path] = None,
    ) -> dict:
        """Walk-forward CV fit on a list of feature DataFrames.

        Returns a report dict with CV accuracy and feature importances.
        """
        import xgboost as xgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score

        # Build combined labelled dataset.
        frames = []
        for df in df_list:
            labeled = self._make_labels(df)
            if labeled is not None and len(labeled) > self.horizon + 20:
                frames.append(labeled)
        if not frames:
            raise ValueError("No usable data after labeling.")

        data = pd.concat(frames, ignore_index=True).dropna()
        X = data[self._feature_cols].values
        y = data["label"].values
        logger.info(
            "DrawdownPredictor: %d samples, %.1f%% positive (drawdown)",
            len(y), y.mean() * 100,
        )

        # Walk-forward cross-validation.
        tscv = TimeSeriesSplit(n_splits=n_splits)
        auc_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            m = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, proba)
                auc_scores.append(auc)
                logger.info("Fold %d AUC: %.3f", fold, auc)

        # Final fit on all data.
        self._model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)

        report = {
            "n_samples":          int(len(y)),
            "positive_rate":      float(y.mean()),
            "cv_mean_auc":        float(np.mean(auc_scores)) if auc_scores else None,
            "cv_auc_scores":      [round(s, 4) for s in auc_scores],
            "feature_importance": dict(
                zip(self._feature_cols,
                    [round(float(v), 4) for v in self._model.feature_importances_])
            ),
        }

        out = Path(save_path or _MODEL_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(out))
        (_FEATURES_PATH).write_text(json.dumps(self._feature_cols))
        logger.info("DrawdownPredictor saved to %s  AUC=%.3f",
                    out, report["cv_mean_auc"] or 0)
        return report

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> float:
        """Return drawdown probability for the most recent bar.

        Returns 0.5 (neutral) if no model is loaded or features are missing.
        """
        if self._model is None:
            return 0.5
        try:
            row = features[self._feature_cols].iloc[-1:].fillna(0.0)
            proba = self._model.predict_proba(row.values)[0, 1]
            return float(np.clip(proba, 0.0, 1.0))
        except Exception as exc:
            logger.debug("DrawdownPredictor.predict_proba failed: %s", exc)
            return 0.5

    def size_multiplier(self, prob: float) -> float:
        """Convert drawdown probability to a position-size multiplier.

        Below _LOW_RISK_PROB  → 1.0  (full size)
        Above _HIGH_RISK_PROB → min_scale
        Between              → linear interpolation
        """
        prob = float(np.clip(prob, 0.0, 1.0))
        if prob <= _LOW_RISK_PROB:
            return 1.0
        if prob >= _HIGH_RISK_PROB:
            return self.min_scale
        t = (prob - _LOW_RISK_PROB) / (_HIGH_RISK_PROB - _LOW_RISK_PROB)
        return float(1.0 - t * (1.0 - self.min_scale))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, path: Optional[Path] = None) -> None:
        import xgboost as xgb

        p = Path(path or _MODEL_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Drawdown model not found: {p}")
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(p))
        if _FEATURES_PATH.exists():
            self._feature_cols = json.loads(_FEATURES_PATH.read_text())
        logger.info("DrawdownPredictor loaded from %s", p)

    @classmethod
    def load_if_exists(cls) -> Optional["DrawdownPredictor"]:
        """Return a loaded instance if the model file exists, else None."""
        if not _MODEL_PATH.exists():
            return None
        try:
            inst = cls()
            inst.load()
            return inst
        except Exception as exc:
            logger.warning("Could not load drawdown model: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_labels(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Add a binary 'label' column: 1 if price drops > dd_threshold in next horizon bars."""
        df = df.copy().reset_index(drop=True)
        missing = [c for c in self._feature_cols + ["close"] if c not in df.columns]
        if missing:
            logger.debug("_make_labels: missing columns %s — skipping", missing)
            return None

        n = len(df)
        labels = np.zeros(n, dtype=int)
        close = df["close"].values
        for i in range(n - self.horizon):
            future_min = close[i + 1: i + 1 + self.horizon].min()
            if future_min < close[i] * (1 - self.dd_threshold):
                labels[i] = 1

        df["label"] = labels
        # Drop last `horizon` rows — their labels peek into future data
        return df.iloc[: n - self.horizon]
