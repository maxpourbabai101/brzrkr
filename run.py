"""trading_enhancer — CLI entry point.

Two modes:

    python run.py --mode live
        Pulls fresh data, runs the ensemble, applies risk filters,
        writes any qualifying signal to data/signals/<ts>.json.

    python run.py --mode backtest --data data/sample.csv
        Replays historical data through the same model + risk stack.

Both modes share the same pipeline so behaviour matches between
research and production.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

# Load .env automatically so API keys don't need to be pre-exported in shell.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass   # dotenv not installed; keys must already be in the environment

# Make `import src.*` work regardless of CWD.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_data
from src.data_scraper import WebDataScraper
from src.features.feature_engineer import FeatureEngineer
from src.signals.signal_generator import generate_signal
from src.utils.logging_setup import configure_logging
from src.execution.broker import AlpacaExecutor

logger = logging.getLogger("trading_enhancer.run")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_config(path: Path = ROOT / "config" / "config.yaml") -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Ensemble construction
# ---------------------------------------------------------------------------
def _build_ensemble():
    """Construct the ensemble. Wire real models in here after training.

    The default factory returns three lightweight heuristic sub-models so
    the pipeline is runnable end-to-end before trained checkpoints exist.
    Each heuristic now consumes both the price DataFrame **and** the
    point-in-time context features attached as ``features.attrs['context']``
    by :class:`FeatureEngineer`. Once real models replace these stubs,
    the same interface (``.predict(features)``) still applies.
    """
    from src.model.ensemble import EnsemblePredictor, EnsembleWeights

    class _PriceMomentum:
        """LSTM stand-in: pure price-trend heuristic."""
        label = "lstm"

        def predict(self, features) -> Dict[str, Any]:
            close = features["close"]
            ret_20 = float(close.pct_change(20).iloc[-1] or 0.0)
            direction = "long" if ret_20 >= 0 else "short"
            edge = abs(ret_20)
            return {
                "direction": direction,
                "expected_return_pct": ret_20,
                "iv_change_pct": 0.0,
                "confidence": float(min(0.5 + 4 * edge, 0.95)),
            }

    class _TabularSentiment:
        """XGBoost stand-in: multi-factor heuristic using price momentum,
        available sentiment, and technical confirmation signals."""
        label = "xgboost"

        def predict(self, features) -> Dict[str, Any]:
            ctx  = features.attrs.get("context", {})
            close = features["close"]

            # Multi-timeframe momentum
            ret_5  = float(close.pct_change(5).iloc[-1]  or 0.0)
            ret_20 = float(close.pct_change(20).iloc[-1] or 0.0)

            # Aggregate sentiment across whichever text sources are present.
            sentiments = []
            for k in ("news_sent_mean", "google_news_sent_mean",
                      "reddit_sent_mean", "stocktwits_sent_mean",
                      "hacker_news_sent_mean", "av_sent_mean",
                      "finnhub_news_sent_mean"):
                v = ctx.get(k)
                if v is not None and not np.isnan(float(v)):
                    sentiments.append(float(v))
            sentiment_score = float(np.mean(sentiments)) if sentiments else 0.0

            # Smart-money: congress net + insider net.
            congress_net = float(ctx.get("congress_net_60d", 0.0) or 0.0)
            insider_net  = float(ctx.get("insider_net_share_change", 0.0) or 0.0)
            smart_money  = float(np.sign(congress_net) + np.sign(insider_net))

            # Volume confirmation: recent vol vs 50d avg (if available)
            vol_conf = 0.0
            try:
                vol = features["volume"]
                vol_ratio = float(vol.iloc[-1] / max(vol.iloc[-50:].mean(), 1))
                if vol_ratio > 1.5 and ret_5 > 0:
                    vol_conf = 0.01   # volume surge on up move = extra bullish
                elif vol_ratio > 1.5 and ret_5 < 0:
                    vol_conf = -0.01
            except Exception:
                pass

            # RSI momentum confirmation (above 50 = trend continuation)
            rsi_conf = 0.0
            try:
                rsi = float(features.get("rsi", pd.Series([50])).iloc[-1] or 50.0)
                rsi_conf = (rsi - 50.0) * 0.0002   # tiny nudge ±0.01 range
            except Exception:
                pass

            # Compose: momentum (60%) + sentiment (25%) + smart-money (5%) +
            #          volume (5%) + rsi (5%)
            composite = (0.35 * ret_5 + 0.25 * ret_20
                         + 0.25 * sentiment_score
                         + 0.05 * smart_money * 0.01
                         + vol_conf + rsi_conf)
            direction = "long" if composite >= 0 else "short"
            edge = abs(composite)
            return {
                "direction": direction,
                "expected_return_pct": composite,
                "iv_change_pct": 0.0,
                "confidence": float(min(0.5 + 4 * edge, 0.95)),
            }

    class _TransformerVol:
        """Transformer stand-in: trend + momentum regime.

        BUG FIX (was: RSI tilt was inverted — RSI > 50 produced a SHORT
        signal, contradicting _PriceMomentum and _TabularSentiment on all
        uptrending stocks, collapsing ensemble confidence to ~0.10-0.22).

        Fixed: RSI acts as a momentum CONFIRMER (RSI > 50 = bullish), with
        extreme readings (>75 / <25) giving very mild mean-reversion nudges.
        """
        label = "transformer"

        def predict(self, features) -> Dict[str, Any]:
            ctx = features.attrs.get("context", {})
            close = features["close"]
            sma_spread = float(features.get("sma_spread", pd.Series([0])).iloc[-1] or 0.0)
            rsi = float(features.get("rsi", pd.Series([50])).iloc[-1] or 50.0)

            # IV change estimate: skew widening = vol pickup expected.
            iv_skew = ctx.get("iv_skew")
            iv_change = (float(iv_skew) * 0.5
                         if iv_skew is not None and not np.isnan(float(iv_skew))
                         else 0.0)

            # RSI momentum tilt (FIXED: momentum interpretation, not mean-reversion).
            # RSI 50-70 = mild bullish confirmation.
            # RSI > 75 = only very slight pullback signal.
            # RSI 30-50 = mild bearish.
            # RSI < 25 = only very slight oversold bounce.
            if rsi >= 75:
                tilt = -0.005   # tiny overbought caution
            elif rsi >= 50:
                tilt = (rsi - 50.0) * 0.0003  # +0 to +0.0075 momentum boost
            elif rsi <= 25:
                tilt = 0.005    # tiny oversold bounce
            else:
                tilt = (rsi - 50.0) * 0.0003  # -0.0075 to 0 bearish lean

            composite = sma_spread + tilt
            direction = "long" if composite >= 0 else "short"
            edge = abs(composite)
            return {
                "direction": direction,
                "expected_return_pct": composite,
                "iv_change_pct": iv_change,
                "confidence": float(min(0.5 + 5 * edge, 0.95)),
            }

    # If a trained XGBoost model exists (produced by `python train.py`),
    # load it as the xgboost sub-model. Otherwise fall back to the
    # _TabularSentiment heuristic.
    from src.model.xgb_predictor import maybe_load_xgb_predictor
    xgb_predictor = maybe_load_xgb_predictor()
    if xgb_predictor is not None:
        xgb_sub = xgb_predictor
        logger.info("Ensemble using TRAINED XGBoost from models/xgb.json")
    else:
        xgb_sub = _TabularSentiment()
        logger.info("Ensemble using HEURISTIC xgboost stand-in (no trained model found). "
                    "Run `python train.py` to train one.")

    return EnsemblePredictor(
        lstm=_PriceMomentum(),
        xgboost=xgb_sub,
        transformer=_TransformerVol(),
        weights=EnsembleWeights(),
    )


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------
def _fetch_bundle(symbol: str, cfg: Dict[str, Any], source: str,
                  *, score_sentiment: bool) -> Dict[str, pd.DataFrame]:
    """Fetch the data bundle from the requested source(s).

    `source` is one of ``"api"`` (load_data only), ``"scraper"``
    (WebDataScraper only), or ``"both"`` (scraper extras merged into
    load_data output).
    """
    if source == "api":
        return load_data(
            symbol,
            lookback_days=cfg["data"]["lookback_days"],
            macro_series=cfg["data"]["macro_series"],
            news_query=cfg["data"]["news_query"],
        )
    if source == "scraper":
        agent = WebDataScraper()
        return agent.scrape_all(symbol, score=score_sentiment, extras=True)
    if source == "both":
        api_bundle = load_data(
            symbol,
            lookback_days=cfg["data"]["lookback_days"],
            macro_series=cfg["data"]["macro_series"],
            news_query=cfg["data"]["news_query"],
        )
        agent = WebDataScraper()
        scrape_bundle = agent.scrape_all(symbol, score=score_sentiment, extras=True)
        # API price feed takes precedence (it's the canonical timeseries).
        merged = dict(scrape_bundle)
        merged.update(api_bundle)
        return merged
    raise ValueError(f"unknown source: {source!r}")


def run_live(
    cfg: Dict[str, Any],
    *,
    execute: bool = False,
    live_money: bool = False,
    source: str = "api",
    score_sentiment: bool = False,
) -> None:
    ensemble = _build_ensemble()
    engineer = FeatureEngineer(window=cfg["model"]["seq_len"])
    universe = cfg["data"]["universe"]
    out_dir = Path(cfg["signals"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    executor: Optional[AlpacaExecutor] = None
    if execute:
        executor = AlpacaExecutor(live_money=live_money)
        equity = executor.get_account_equity()
        logger.info("Execution ENABLED — broker equity: $%.2f", equity)
    else:
        logger.info("Execution disabled — signals will be written but no orders sent.")

    logger.info("Data source: %s  (sentiment scoring: %s)", source, score_sentiment)

    for symbol in universe:
        try:
            bundle = _fetch_bundle(symbol, cfg, source, score_sentiment=score_sentiment)
        except Exception as exc:  # noqa: BLE001
            logger.error("Data load failed for %s: %s", symbol, exc)
            continue

        prices = bundle.get("prices")
        if prices is None or prices.empty or len(prices) < cfg["model"]["seq_len"]:
            n = 0 if prices is None else len(prices)
            logger.warning("%s: insufficient history (%d bars)", symbol, n)
            continue

        features = engineer.build_features(bundle)
        if features.empty:
            logger.warning("%s: feature engineering produced empty frame", symbol)
            continue

        prediction = ensemble.predict(features)

        # Use real account equity when executing; fall back to config otherwise.
        account_equity = (
            executor.get_account_equity() if executor is not None
            else cfg["live"]["initial_account_equity_usd"]
        )
        ctx = features.attrs.get("context", {})
        risk_params = {
            "account_equity": account_equity,
            "entry_price": float(prices["close"].iloc[-1]),
            "atr": float((prices["high"] - prices["low"]).tail(14).mean()),
            "vix": float(ctx.get("vix_level") or 0.0),
            "realized_vol": float(prices["close"].pct_change().tail(20).std() or 0.0),
            "current_time": datetime.now(timezone.utc),
            "existing_positions": [],
            "correlation_matrix": {},
        }

        signal = generate_signal(symbol, prediction, risk_params)
        if signal is None:
            continue
        # Attach a compact feature snapshot so the JSON signal records
        # what context drove the trade.
        signal["context_features"] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) and not np.isnan(v) else None)
            for k, v in ctx.items()
        }
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = out_dir / f"{symbol}_{ts}.json"
        path.write_text(json.dumps(signal, indent=2, default=str))
        logger.info("Signal written: %s", path)

        if executor is not None:
            result = executor.submit_signal(signal)
            if result.submitted:
                logger.info("Order submitted: %s (id=%s)", symbol, result.order_id)
            else:
                logger.warning("Order NOT submitted for %s: %s", symbol, result.reason)


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------
def run_backtest(cfg: Dict[str, Any], data_path: str) -> None:
    from src.backtest.backtest_runner import BacktestRunner

    ensemble = _build_ensemble()

    def predict_fn(window):
        return ensemble.predict(window)

    runner = BacktestRunner(
        predict_fn=predict_fn,
        initial_equity=cfg["live"]["initial_account_equity_usd"],
        confidence_threshold=cfg["signals"]["confidence_threshold"],
    )
    history = runner.load_history(data_path)
    result = runner.run(history, window=cfg["model"]["seq_len"])
    logger.info("Backtest summary: %s", result.summary)
    print(json.dumps(result.summary, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="trading_enhancer entry point")
    parser.add_argument("--mode", choices=["live", "backtest"], required=True)
    parser.add_argument("--data", help="Path to a CSV/Parquet file (backtest mode).")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="LIVE MODE ONLY: submit bracket orders to Alpaca (paper by default).",
    )
    parser.add_argument(
        "--live-money",
        action="store_true",
        help=("DANGER: route orders to the REAL‑MONEY Alpaca endpoint. "
              "Requires --execute AND ALPACA_LIVE=true in the environment."),
    )
    parser.add_argument(
        "--source",
        choices=["api", "scraper", "both"],
        default="api",
        help=("Where to pull data from. 'api' = paid/authed vendors "
              "(default). 'scraper' = pure web scraping (no secondary "
              "API keys). 'both' = api primary, scraper extras (Reddit, "
              "Congress, Wiki) merged in."),
    )
    parser.add_argument(
        "--score-sentiment",
        action="store_true",
        help=("Run FinBERT over every text source the loader produces. "
              "Adds ~30s for first-call model load. Has no effect with "
              "--source api."),
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    configure_logging(
        level=cfg["logging"]["level"],
        log_path=cfg["logging"]["log_path"],
    )
    logger.info("Running in %s mode; cwd=%s", args.mode, os.getcwd())

    if args.mode == "live":
        if args.live_money and not args.execute:
            parser.error("--live-money requires --execute")
        run_live(
            cfg,
            execute=args.execute,
            live_money=args.live_money,
            source=args.source,
            score_sentiment=args.score_sentiment,
        )
    else:
        if args.execute or args.live_money:
            parser.error("--execute / --live-money only apply in --mode live")
        if not args.data:
            parser.error("--data is required for backtest mode")
        run_backtest(cfg, args.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
