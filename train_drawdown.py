"""train_drawdown.py — Train the BRZRKR drawdown predictor.

Downloads historical price data, engineers features, trains an XGBoost
classifier to predict near-term drawdowns, and saves the model to
models/drawdown_model.json.

Usage
-----
    python train_drawdown.py
    python train_drawdown.py --symbols SPY QQQ AAPL MSFT --dd-threshold 0.02
    python train_drawdown.py --horizon 3 --lookback-days 1000

The next agent.py / run.py invocation auto-loads models/drawdown_model.json
and applies the drawdown probability as a position-size multiplier.
Remove the model to revert: rm models/drawdown_model.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _fetch_features(symbol: str, lookback_days: int):
    """Download OHLCV and run FeatureEngineer. Returns None on failure."""
    try:
        import yfinance as yf
        import pandas as pd
        from src.features.feature_engineer import FeatureEngineer

        raw = yf.download(symbol, period=f"{lookback_days}d", interval="1d",
                          auto_adjust=False, progress=False)
        if raw.empty:
            logger.warning("No data for %s", symbol)
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.rename(columns={"adj close": "close"}).reset_index()
        fe  = FeatureEngineer()
        df  = fe.build_features({"prices": raw}, full_history=True)
        if df.empty or len(df) < 60:
            logger.warning("Too few bars for %s (%d)", symbol, len(df))
            return None
        logger.info("Fetched %d bars for %s", len(df), symbol)
        return df
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", symbol, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BRZRKR drawdown predictor")
    parser.add_argument("--symbols", nargs="+",
                        default=["SPY", "QQQ", "IWM", "DIA", "AAPL",
                                 "MSFT", "NVDA", "AMD", "TSLA"],
                        help="Symbols for training data (default: broad market mix)")
    parser.add_argument("--lookback-days", type=int, default=1000,
                        help="Days of history per symbol (default 1000)")
    parser.add_argument("--dd-threshold", type=float, default=0.015,
                        help="Drawdown threshold to label as 1 (default 1.5%%)")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Look-ahead bars for labeling (default 5)")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Walk-forward CV folds (default 5)")
    parser.add_argument("--output", type=str, default="models/drawdown_model.json")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logger.info("Fetching data for: %s", args.symbols)
    df_list = [_fetch_features(sym, args.lookback_days) for sym in args.symbols]
    df_list = [df for df in df_list if df is not None]

    if not df_list:
        logger.error("No usable data — check internet connection.")
        sys.exit(1)

    logger.info(
        "Training drawdown predictor: %d symbols, horizon=%d, threshold=%.1f%%",
        len(df_list), args.horizon, args.dd_threshold * 100,
    )

    from src.risk.drawdown_predictor import DrawdownPredictor
    model = DrawdownPredictor(
        dd_threshold=args.dd_threshold,
        horizon=args.horizon,
    )
    report = model.train(
        df_list,
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        save_path=Path(args.output),
    )

    print("\nDrawdown predictor training complete")
    print(f"  Samples:       {report['n_samples']}")
    print(f"  Positive rate: {report['positive_rate']:.1%}  (drawdown events)")
    auc = report.get("cv_mean_auc")
    print(f"  CV mean AUC:   {auc:.3f}" if auc else "  CV AUC:        n/a")
    print(f"\nTop features:")
    for feat, imp in sorted(report["feature_importance"].items(),
                             key=lambda x: -x[1])[:5]:
        print(f"  {feat:20s}  {imp:.4f}")
    print(f"\nModel saved to: {args.output}")
    print("Position sizes will be automatically scaled on the next agent.py run.")


if __name__ == "__main__":
    main()
