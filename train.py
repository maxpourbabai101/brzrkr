"""Training CLI — fit an XGBoost classifier on FeatureEngineer output.

Quick start:
    python train.py                                # SPY, 2y history, default labels
    python train.py --symbols SPY QQQ AAPL --lookback-days 1095
    python train.py --horizon 10 --long-threshold 0.01 --short-threshold 0.01

Outputs (all under `models/`):
    xgb.json           — XGBoost model
    xgb_features.json  — ordered feature column names
    xgb_report.json    — CV scores, feature importance, label distribution

After training, the next `python run.py --mode live` (or `agent.py`)
will auto-detect the saved model and use it in place of the
heuristic XGBoost sub-model.

Honest framing
--------------
Walk-forward CV accuracy of 0.52–0.55 is "expected" for this kind of
short-horizon classifier on liquid US equities. Anything dramatically
above that on cheaply-engineered features is almost certainly leakage
or overfitting — read the "READ FIRST" section of
docs/trading_strategy_sources.md before celebrating any model that
looks too good.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_engineer import FeatureEngineer
from src.learning.preflight import run_preflight
from src.training.feature_dataset import FeatureDataset, LabelConfig
from src.training.trainer import TrainConfig, XGBoostTrainer
from src.utils.logging_setup import configure_logging

logger = logging.getLogger("trading_enhancer.train")


# ---------------------------------------------------------------------------
# Data fetcher — tries Alpaca first, falls back to Yahoo via scraper
# ---------------------------------------------------------------------------
def _fetch_prices(
    symbol: str,
    lookback_days: int,
    *,
    prefer_scraper: bool = False,
) -> pd.DataFrame:
    """Return a datetime-indexed OHLCV DataFrame for `symbol`.

    If ``prefer_scraper`` is True, skips Alpaca and goes straight to
    Yahoo via the scraper — much faster for bulk historical fetches.
    """
    if not prefer_scraper:
        # Try Alpaca first (cleaner, requires keys).
        try:
            from src.data_loader import fetch_futures
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            df = fetch_futures(symbol, start, end)
            if not df.empty and len(df) > 50:
                return df
        except Exception as exc:  # noqa: BLE001
            logger.info("Alpaca fetch failed for %s (%s); trying scraper.", symbol, exc)

    # Yahoo via the scraper (much faster for bulk).
    from src.data_scraper import WebDataScraper
    agent = WebDataScraper()
    df = agent.scrape_ohlcv(
        symbol,
        range_=_lookback_to_yahoo_range(lookback_days),
    )
    if df.empty:
        raise RuntimeError(f"Could not fetch any price history for {symbol}")
    return df


def _lookback_to_yahoo_range(days: int) -> str:
    if days <= 30: return "1mo"
    if days <= 90: return "3mo"
    if days <= 180: return "6mo"
    if days <= 365: return "1y"
    if days <= 730: return "2y"
    if days <= 1825: return "5y"
    return "10y"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost on FeatureEngineer output")
    parser.add_argument("--symbols", nargs="+", default=["SPY"],
                        help="One or more tickers to pool into the training set.")
    parser.add_argument("--lookback-days", type=int, default=730,
                        help="History per symbol (default 730 = 2y).")

    # Labeling.
    parser.add_argument("--horizon", type=int, default=5,
                        help="Forward-return horizon in bars (default 5).")
    parser.add_argument("--long-threshold", type=float, default=0.005,
                        help="Forward return above this → long label (default 0.5%%).")
    parser.add_argument("--short-threshold", type=float, default=0.005,
                        help="Forward return below this (abs) → short label (default 0.5%%).")

    # Training.
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Walk-forward CV folds (default 5).")
    parser.add_argument("--test-size", type=int, default=60,
                        help="Bars per CV test fold (default 60).")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=4)

    parser.add_argument("--output-dir", default="models",
                        help="Where to save xgb.json + features + report.")
    parser.add_argument("--prefer-scraper", action="store_true",
                        help=("Skip Alpaca and fetch from Yahoo via the scraper. "
                              "Much faster for bulk historical pulls."))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    configure_logging(level=args.log_level, log_path="trading_enhancer.log")

    # Preflight: consult the postmortem DB before spending compute on a
    # train that may already be known to fail in a documented way.
    pre = run_preflight("train")
    print(pre.to_text())

    label_cfg = LabelConfig(
        horizon_bars=args.horizon,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )
    fe = FeatureEngineer()
    ds = FeatureDataset(feature_engineer=fe, label_cfg=label_cfg)

    all_X: List[pd.DataFrame] = []
    all_y: List[pd.Series] = []
    total = len(args.symbols)
    print(f"\nFetching {total} symbol(s) × {args.lookback_days} days...")
    print("(Alpaca REST is slow for bulk data; pass --prefer-scraper for Yahoo speed)\n")
    for i, symbol in enumerate(args.symbols, 1):
        print(f"  [{i}/{total}] {symbol}: fetching...", end="", flush=True)
        t0 = datetime.now()
        try:
            prices = _fetch_prices(symbol, args.lookback_days,
                                    prefer_scraper=args.prefer_scraper)
        except Exception as exc:  # noqa: BLE001
            print(f" ✗ FAILED ({exc})")
            continue
        dt = (datetime.now() - t0).total_seconds()
        print(f" got {len(prices)} bars in {dt:.1f}s; building features...",
              end="", flush=True)
        try:
            X, y = ds.build(prices)
        except Exception as exc:  # noqa: BLE001
            print(f" ✗ dataset build failed ({exc})")
            continue
        print(f" {len(X)} samples ✓")
        all_X.append(X)
        all_y.append(y)
    print()

    if not all_X:
        logger.error("No training data produced. Aborting.")
        return 2

    # Pool across symbols. They must share columns.
    common_cols = set.intersection(*(set(x.columns) for x in all_X))
    X = pd.concat([x[sorted(common_cols)] for x in all_X], ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    logger.info("Combined training set: %d samples × %d features",
                X.shape[0], X.shape[1])

    trainer = XGBoostTrainer(TrainConfig(
        n_splits=args.n_splits,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    ))
    report = trainer.train(X, y, output_dir=Path(args.output_dir))

    print()
    print(f"=== Training complete ===")
    print(f"Samples            : {report.n_samples}")
    print(f"Features           : {report.n_features}")
    print(f"Label distribution : {report.label_distribution}")
    print(f"Mean CV accuracy   : {report.mean_cv_accuracy:.4f}")
    print(f"Mean CV log-loss   : {report.mean_cv_logloss:.4f}")
    print(f"Per-fold accuracy  : "
          + ", ".join(f"{s:.4f}" for s in report.cv_scores))
    print(f"\nTop 10 features by gain:")
    for k, v in list(report.feature_importance.items())[:10]:
        print(f"  {k:30s}  {v:.2f}")
    print()
    print(f"Artifacts written to: {args.output_dir}/")
    print(f"Next run.py / agent.py will auto-load this model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
