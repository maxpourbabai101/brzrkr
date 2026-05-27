"""XGBoostTrainer — walk-forward CV + final fit, saves to models/.

Walk-forward (vs. random KFold) is essential in finance: random
shuffling leaks future information into training and inflates scores
catastrophically. We use ``sklearn.model_selection.TimeSeriesSplit``
which preserves temporal order.

Output:
    models/xgb.json           — XGBoost model (JSON format)
    models/xgb_features.json  — ordered list of feature column names
    models/xgb_report.json    — CV scores + metadata

The inference wrapper :class:`src.model.xgb_predictor.XGBoostPredictor`
expects all three files to exist together.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    n_splits: int = 5                # walk-forward CV folds
    test_size: int = 60              # bars per test fold
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    early_stopping_rounds: Optional[int] = 25
    random_state: int = 42


@dataclass
class TrainReport:
    cv_scores: List[float] = field(default_factory=list)
    cv_logloss: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_features: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    trained_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def mean_cv_accuracy(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else 0.0

    @property
    def mean_cv_logloss(self) -> float:
        return float(np.mean(self.cv_logloss)) if self.cv_logloss else 0.0


class XGBoostTrainer:
    def __init__(self, cfg: Optional[TrainConfig] = None) -> None:
        self.cfg = cfg or TrainConfig()

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: Path = Path("models"),
    ) -> TrainReport:
        # Lazy import so the rest of the project loads without xgboost.
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.model_selection import TimeSeriesSplit
        import xgboost as xgb

        if X.empty or y.empty:
            raise ValueError("Empty training data.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")

        cfg = self.cfg
        report = TrainReport(
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            label_distribution={str(k): int(v) for k, v in y.value_counts().items()},
            trained_at=datetime.now(timezone.utc).isoformat(),
        )

        # Auto-adjust CV parameters if the dataset is too small.
        # TimeSeriesSplit requires n_samples >= (n_splits + 1) * test_size.
        n_splits = cfg.n_splits
        test_size = cfg.test_size
        n_samples = X.shape[0]
        required = (n_splits + 1) * test_size
        if required > n_samples:
            test_size = max(10, n_samples // (n_splits + 1))
            if (n_splits + 1) * test_size > n_samples:
                n_splits = max(2, (n_samples // test_size) - 1)
            logger.warning(
                "CV auto-adjusted for small dataset: n_splits=%d→%d, "
                "test_size=%d→%d (n_samples=%d). "
                "For more robust validation, expand --symbols or --lookback-days.",
                cfg.n_splits, n_splits, cfg.test_size, test_size, n_samples,
            )

        # Walk-forward CV.
        splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            model = self._make_estimator(xgb)
            fit_kwargs = {"verbose": False}
            if cfg.early_stopping_rounds:
                fit_kwargs["eval_set"] = [(X_te, y_te)]
            model.fit(X_tr, y_tr, **fit_kwargs)
            preds = model.predict(X_te)
            proba = model.predict_proba(X_te)
            acc = accuracy_score(y_te, preds)
            ll = log_loss(y_te, proba, labels=[0, 1])
            report.cv_scores.append(float(acc))
            report.cv_logloss.append(float(ll))
            logger.info("CV fold %d/%d: accuracy=%.4f logloss=%.4f",
                        fold + 1, n_splits, acc, ll)

        # Final fit on everything. Disable early stopping here — there's
        # no eval set when training on all data. (XGBoost ≥ 2.0 raises
        # ValueError if early_stopping_rounds is set without an eval_set.)
        final = self._make_estimator(xgb, allow_early_stopping=False)
        final.fit(X, y, verbose=False)
        booster = final.get_booster()
        # Per-feature importance (gain).
        try:
            raw_imp = booster.get_score(importance_type="gain")
            # XGBoost uses f0/f1/... by default. Map back to column names.
            mapped = {}
            for k, v in raw_imp.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if 0 <= idx < X.shape[1]:
                        mapped[X.columns[idx]] = float(v)
                else:
                    mapped[k] = float(v)
            # Sort descending.
            report.feature_importance = dict(
                sorted(mapped.items(), key=lambda kv: kv[1], reverse=True)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not extract feature importance: %s", exc)

        # Persist.
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        final.save_model(str(output_dir / "xgb.json"))
        (output_dir / "xgb_features.json").write_text(
            json.dumps(list(X.columns), indent=2)
        )
        (output_dir / "xgb_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str)
        )
        logger.info(
            "Saved trained model to %s (mean CV accuracy %.4f, mean CV log-loss %.4f)",
            output_dir, report.mean_cv_accuracy, report.mean_cv_logloss,
        )
        return report

    def _make_estimator(self, xgb_module, *, allow_early_stopping: bool = True):
        cfg = self.cfg
        kwargs = dict(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=cfg.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
        if allow_early_stopping and cfg.early_stopping_rounds:
            kwargs["early_stopping_rounds"] = cfg.early_stopping_rounds
        return xgb_module.XGBClassifier(**kwargs)
