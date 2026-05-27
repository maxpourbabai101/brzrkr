"""OnlineLearner — incremental model retraining after every closed trade.

After enough new trades accumulate in the TradeJournal the learner:

1. Builds a feature matrix from all closed trades (features recorded
   at signal time, label = 1 if win, 0 if loss).
2. Fine-tunes the resident XGBoost model with ``update`` (one extra
   boosting round on the new data).
3. Saves the updated model back to ``models/xgb.json``.
4. Recalibrates the confidence threshold based on the model's recent
   precision on the journal data.
5. Writes a learning report to ``data/learning_report.json`` so the
   UI and logs can surface what changed.

The learner is designed to be called from the agent after each
completed tick where a position was closed.  It is safe to call
frequently — it debounces itself (won't retrain if fewer than
``min_new_trades`` closed trades arrived since the last run).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_PATH      = Path("models/xgb.json")
_FEATURES_PATH   = Path("models/xgb_features.json")
_REPORT_PATH     = Path("data/learning_report.json")
_STATE_PATH      = Path("data/learner_state.json")


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------

class OnlineLearner:
    """Self-improving wrapper around the XGBoost model.

    Parameters
    ----------
    min_new_trades : int
        Minimum number of *new* closed trades required since the last
        retraining run before we bother fitting.
    lookback : int
        How many historical closed trades to include in each refit
        (sliding window to prevent very old data dominating).
    """

    def __init__(
        self,
        min_new_trades: int = 5,
        lookback: int = 200,
        model_path: Path = _MODEL_PATH,
        features_path: Path = _FEATURES_PATH,
    ) -> None:
        self.min_new_trades = min_new_trades
        self.lookback       = lookback
        self.model_path     = Path(model_path)
        self.features_path  = Path(features_path)
        self._state         = self._load_state()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def maybe_update(self) -> Optional[Dict[str, Any]]:
        """Check if enough new trades exist and retrain if so.

        Returns a report dict if retraining happened, None otherwise.
        """
        try:
            from src.learning.trade_journal import TradeJournal
            journal  = TradeJournal()
            closed   = journal.get_closed_trades()
        except Exception as exc:
            logger.warning("OnlineLearner: could not read journal — %s", exc)
            return None

        last_seen = self._state.get("last_seen_trade_count", 0)
        if len(closed) - last_seen < self.min_new_trades:
            return None  # not enough new data yet

        return self._retrain(closed)

    def force_update(self) -> Optional[Dict[str, Any]]:
        """Retrain unconditionally on all available closed trades."""
        try:
            from src.learning.trade_journal import TradeJournal
            closed = TradeJournal().get_closed_trades()
        except Exception as exc:
            logger.warning("OnlineLearner: journal read failed — %s", exc)
            return None
        if not closed:
            return None
        return self._retrain(closed)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _retrain(self, closed: List[Dict]) -> Optional[Dict[str, Any]]:
        """Core retraining logic."""
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed — skipping online update")
            return None

        feature_names = self._load_feature_names()
        if not feature_names:
            logger.warning("OnlineLearner: no feature names found — skipping")
            return None

        # Build dataset from the last `lookback` closed trades that have
        # feature snapshots stored.
        window = closed[-self.lookback:]
        rows, labels = [], []
        for trade in window:
            feats = trade.get("features")
            if not feats:
                continue
            label = 1 if trade.get("outcome") == "win" else 0
            row = [float(feats.get(f, 0.0)) for f in feature_names]
            if any(math.isnan(v) or math.isinf(v) for v in row):
                continue
            rows.append(row)
            labels.append(label)

        if len(rows) < self.min_new_trades:
            logger.info("OnlineLearner: only %d usable rows — skipping", len(rows))
            return None

        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        dmat = xgb.DMatrix(X, label=y, feature_names=feature_names)

        # Load existing model and run one update round.
        booster = None
        if self.model_path.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(self.model_path))
            except Exception as exc:
                logger.warning("Could not load model for update: %s", exc)
                booster = None

        params = {
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "max_depth":        4,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "tree_method":      "hist",
            "verbosity":        0,
        }

        n_rounds = 5  # fine-tune with a few extra trees
        if booster is not None:
            # Continue training (append trees)
            booster = xgb.train(params, dmat,
                                num_boost_round=n_rounds,
                                xgb_model=booster,
                                verbose_eval=False)
        else:
            # Fresh start if model file was missing or corrupt
            booster = xgb.train(params, dmat,
                                num_boost_round=50,
                                verbose_eval=False)

        # Save updated model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(self.model_path))

        # Compute in-sample metrics for the report
        preds    = booster.predict(dmat)
        pred_bin = (preds >= 0.5).astype(int)
        accuracy = float(np.mean(pred_bin == y))
        wins     = int(np.sum(y == 1))
        losses   = int(np.sum(y == 0))
        win_rate = float(wins / max(len(y), 1))

        # Suggest a recalibrated confidence threshold:
        # Use the precision-recall tradeoff — find threshold where
        # precision (= wins among predicted wins) ≥ 0.60.
        threshold = self._calibrate_threshold(preds, y)

        report = {
            "retrained_at":   datetime.now(timezone.utc).isoformat(),
            "training_rows":  len(rows),
            "total_closed":   len(closed),
            "win_rate":       round(win_rate, 3),
            "in_sample_acc":  round(accuracy, 3),
            "suggested_confidence_threshold": round(threshold, 3),
            "feature_count":  len(feature_names),
            "wins":           wins,
            "losses":         losses,
        }

        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(json.dumps(report, indent=2))

        # Persist state so we know where we left off next run.
        self._state["last_seen_trade_count"] = len(closed)
        self._state["last_retrain"]          = report["retrained_at"]
        self._state["confidence_threshold"]  = threshold

        # Append to accuracy history for UI display
        history = self._state.get("accuracy_history", [])
        history.append({
            "ts":            report["retrained_at"],
            "training_rows": len(rows),
            "in_sample_acc": round(accuracy, 3),
            "win_rate":      round(win_rate, 3),
            "confidence_threshold": round(threshold, 3),
        })
        self._state["accuracy_history"] = history[-50:]   # keep last 50 updates
        self._save_state()

        logger.info(
            "OnlineLearner: retrained on %d rows — acc=%.2f  "
            "win_rate=%.2f  suggested_conf_threshold=%.2f",
            len(rows), accuracy, win_rate, threshold,
        )
        return report

    def _calibrate_threshold(
        self, probs: np.ndarray, labels: np.ndarray,
        target_precision: float = 0.60,
    ) -> float:
        """Find lowest threshold where precision >= target_precision."""
        best_thresh = 0.75  # safe default
        for t in np.arange(0.50, 0.95, 0.01):
            mask = probs >= t
            if mask.sum() < 5:
                break
            precision = labels[mask].mean()
            if precision >= target_precision:
                best_thresh = float(t)
                break
        return best_thresh

    def _load_feature_names(self) -> List[str]:
        if self.features_path.exists():
            try:
                data = json.loads(self.features_path.read_text())
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("feature_names", [])
            except Exception:
                pass
        # Fallback: try reading from the model itself
        if self.model_path.exists():
            try:
                import xgboost as xgb
                b = xgb.Booster()
                b.load_model(str(self.model_path))
                names = b.feature_names
                if names:
                    return names
            except Exception:
                pass
        return []

    def get_suggested_threshold(self) -> Optional[float]:
        """Return the last calibrated confidence threshold, if any."""
        return self._state.get("confidence_threshold")

    def _load_state(self) -> Dict[str, Any]:
        if _STATE_PATH.exists():
            try:
                return json.loads(_STATE_PATH.read_text())
            except Exception:
                pass
        return {}

    def _save_state(self) -> None:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(self._state, indent=2))


# ---------------------------------------------------------------------------
# Convenience function — used by the agent tick loop
# ---------------------------------------------------------------------------

_learner: Optional[OnlineLearner] = None


def get_learner() -> OnlineLearner:
    global _learner
    if _learner is None:
        _learner = OnlineLearner()
    return _learner
