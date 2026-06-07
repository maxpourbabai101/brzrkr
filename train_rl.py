"""train_rl.py — Train the BRZRKR reinforcement learning agent.

Downloads historical price data, runs it through FeatureEngineer,
trains a PPO agent, and saves the model to models/rl_ppo.zip.

Usage
-----
    python train_rl.py
    python train_rl.py --symbols SPY QQQ AAPL NVDA --timesteps 500000
    python train_rl.py --lookback-days 365 --window 30 --timesteps 100000

The next agent.py / run.py invocation will auto-detect models/rl_ppo.zip
and include the RL agent as a fourth ensemble member. To remove it:

    rm models/rl_ppo.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _fetch_df(symbol: str, lookback_days: int):
    """Download OHLCV and engineer features. Returns None on failure."""
    try:
        import yfinance as yf
        import pandas as pd
        from src.features.feature_engineer import FeatureEngineer

        period = f"{lookback_days}d"
        raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False,
                          progress=False)
        if raw.empty:
            logger.warning("No data for %s", symbol)
            return None

        # Flatten MultiIndex if present (yfinance ≥ 0.2)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.rename(columns={"adj close": "close"}).reset_index()
        bundle = {"prices": raw}
        fe = FeatureEngineer()
        df = fe.build_features(bundle, full_history=True)
        if df.empty or len(df) < 50:
            logger.warning("Not enough data for %s (%d rows)", symbol, len(df))
            return None
        logger.info("Fetched %d bars for %s", len(df), symbol)
        return df
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", symbol, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BRZRKR RL agent (PPO)")
    parser.add_argument("--symbols", nargs="+",
                        default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD"],
                        help="Ticker symbols for training data")
    parser.add_argument("--lookback-days", type=int, default=730,
                        help="Days of history to pull per symbol (default 730)")
    parser.add_argument("--window", type=int, default=20,
                        help="Observation window in bars (default 20)")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="PPO training timesteps (default 300 000)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel training environments (default 4)")
    parser.add_argument("--output", type=str, default="models/rl_ppo",
                        help="Model output path (default models/rl_ppo)")
    args = parser.parse_args()

    # Try to load dotenv for API keys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logger.info("Fetching training data for: %s", args.symbols)
    df_list = [_fetch_df(sym, args.lookback_days) for sym in args.symbols]
    df_list = [df for df in df_list if df is not None]

    if not df_list:
        logger.error("No data fetched — aborting. Check your internet connection.")
        sys.exit(1)

    logger.info("Training on %d symbol(s), %d timesteps", len(df_list), args.timesteps)

    from src.rl.rl_agent import RLAgent
    agent = RLAgent()
    agent.train(
        df_list,
        window=args.window,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_path=Path(args.output),
        verbose=1,
    )

    print(f"\nRL agent trained and saved to {args.output}.zip")
    print("It will be loaded automatically on the next agent.py / run.py invocation.")
    print("To evaluate:")
    print("  python agent.py --dry-run")


if __name__ == "__main__":
    main()
