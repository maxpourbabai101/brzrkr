"""XGBoostPredictor — inference wrapper matching the ensemble sub-model interface.

Loads ``models/xgb.json`` + ``models/xgb_features.json`` produced by
:class:`src.training.trainer.XGBoostTrainer`, and exposes the same
``.predict(features) -> dict`` interface as the heuristic sub-models
in ``run.py``. Slots directly into :class:`EnsemblePredictor`.

If the model files don't exist, :func:`maybe_load_xgb_predictor`
returns ``None`` so callers can fall back to a heuristic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/xgb.json")
DEFAULT_FEATURES_PATH = Path("models/xgb_features.json")


@dataclass
class XGBoostPredictor:
    model_path: Path = DEFAULT_MODEL_PATH
    features_path: Path = DEFAULT_FEATURES_PATH

    def __post_init__(self) -> None:
        import xgboost as xgb
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file missing: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file missing: {self.features_path}")
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(self.model_path))
        self.feature_names: List[str] = json.loads(self.features_path.read_text())
        logger.info("Loaded XGBoost predictor: %d features from %s",
                    len(self.feature_names), self.model_path)

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        if features is None or features.empty:
            return _abstain()

        # Pull the last row's features matching the trained column order.
        last = features.iloc[-1]
        row = pd.Series(
            {col: float(last.get(col, np.nan)) for col in self.feature_names}
        )
        # If too many NaNs (model was trained on different features), abstain.
        if row.isna().sum() > len(row) * 0.5:
            logger.warning("XGBoostPredictor: >50%% missing features — abstaining")
            return _abstain()
        x = row.fillna(0.0).values.reshape(1, -1)

        try:
            proba = self._model.predict_proba(x)[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("XGBoost predict_proba failed: %s — abstaining", exc)
            return _abstain()

        p_long = float(proba[1]) if len(proba) > 1 else float(proba[0])
        direction = "long" if p_long >= 0.5 else "short"
        confidence = max(p_long, 1.0 - p_long)
        # Pseudo expected return: probability-weighted nudge centered at zero.
        edge = (p_long - 0.5) * 2.0   # in [-1, 1]
        expected_return_pct = float(edge * 0.005)  # 0.5% magnitude scaling

        return {
            "direction": direction,
            "expected_return_pct": expected_return_pct,
            "iv_change_pct": 0.0,
            "confidence": float(confidence),
        }


def _abstain() -> Dict[str, Any]:
    return {
        "direction": "long",
        "expected_return_pct": 0.0,
        "iv_change_pct": 0.0,
        "confidence": 0.5,    # exactly at the no-trade boundary
    }


def maybe_load_xgb_predictor(
    model_path: Path = DEFAULT_MODEL_PATH,
    features_path: Path = DEFAULT_FEATURES_PATH,
) -> Optional[XGBoostPredictor]:
    """Return a predictor if the model files exist; None otherwise."""
    if not (model_path.exists() and features_path.exists()):
        return None
    try:
        return XGBoostPredictor(model_path=model_path, features_path=features_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load XGBoost model from %s: %s",
                       model_path, exc)
        return None
