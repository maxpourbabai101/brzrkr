"""Autonomous trading agent — long-running loop.

Wires together every piece (loader/scraper → feature engineer →
ensemble → risk → broker) into a single continuously-running process.

Quick start (paper, dry-run, scraper-only data):
    python agent.py --dry-run

Real paper trading:
    python agent.py --execute

Real money (requires --live-money AND ALPACA_LIVE=true in env):
    export ALPACA_LIVE=true
    python agent.py --execute --live-money

Stop cleanly:
    Ctrl-C   (finishes the current tick, then exits)
    OR
    touch AGENT_STOP   (detected at the top of the next tick)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Load .env automatically so API keys don't need to be pre-exported in shell.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass   # dotenv not installed; keys must already be in the environment

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.trading_agent import AgentConfig, TradingAgent
from src.data_loader import load_data
from src.data_scraper import WebDataScraper
from src.execution.brokers import get_broker
from src.features.feature_engineer import FeatureEngineer
from src.learning.preflight import run_preflight
from src.signals.signal_generator import generate_signal
from src.utils.logging_setup import configure_logging

# Reuse the ensemble factory from run.py so we don't drift.
from run import _build_ensemble

logger = logging.getLogger("trading_enhancer.agent")


# ---------------------------------------------------------------------------
# Data-source factories
# ---------------------------------------------------------------------------
def _make_data_fetcher(cfg: Dict[str, Any], source: str, score_sentiment: bool):
    """Return a callable `symbol -> bundle` honouring the chosen source."""
    if source == "api":
        return lambda s: load_data(
            s,
            lookback_days=cfg["data"]["lookback_days"],
            macro_series=cfg["data"]["macro_series"],
            news_query=cfg["data"]["news_query"],
        )

    if source == "scraper":
        agent = WebDataScraper()
        return lambda s: agent.scrape_all(s, score=score_sentiment, extras=True)

    if source == "both":
        scraper_agent = WebDataScraper()

        def _both(symbol: str) -> Dict[str, Any]:
            api_bundle = load_data(
                symbol,
                lookback_days=cfg["data"]["lookback_days"],
                macro_series=cfg["data"]["macro_series"],
                news_query=cfg["data"]["news_query"],
            )
            scrape_bundle = scraper_agent.scrape_all(
                symbol, score=score_sentiment, extras=True
            )
            merged = dict(scrape_bundle)
            merged.update(api_bundle)  # API price feed wins
            return merged

        return _both

    raise ValueError(f"unknown source: {source!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous trading agent")
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument(
        "--source", choices=["api", "scraper", "both"], default="api",
        help="Where to pull data from. Same semantics as run.py.",
    )
    parser.add_argument("--score-sentiment", action="store_true",
                        help="Run FinBERT sentiment over scraped text sources.")

    # Loop tuning.
    parser.add_argument("--tick-seconds", type=int, default=300,
                        help="Seconds between ticks (default 300 = 5 min).")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Cap on simultaneous open positions.")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.03,
                        help="Halt for the session at this drawdown (default 3%%).")
    parser.add_argument("--pre-close-minutes", type=int, default=15,
                        help="No new entries within this many minutes of close.")

    # Execution flags (mirror run.py).
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--execute", action="store_true",
                     help="Submit real bracket orders to Alpaca (paper by default).")
    grp.add_argument("--dry-run", action="store_true",
                     help="Run the loop but log signals only; never submit orders.")
    parser.add_argument("--live-money", action="store_true",
                        help=("DANGER: live-money endpoint. Requires --execute "
                              "AND ALPACA_LIVE=true in env AND a passing "
                              "promotion-gate track record."))
    parser.add_argument("--broker", choices=["alpaca", "ibkr", "paper_only"],
                        default="alpaca",
                        help=("Which broker to route orders through. "
                              "`paper_only` = in-memory simulator (no network). "
                              "`ibkr` = Interactive Brokers (stub — implement "
                              "before using). `alpaca` (default) = both paper "
                              "and live, governed by --live-money + ALPACA_LIVE."))

    args = parser.parse_args()

    if args.live_money and not args.execute:
        parser.error("--live-money requires --execute")

    cfg = yaml.safe_load(Path(args.config).read_text())
    configure_logging(
        level=cfg["logging"]["level"],
        log_path=cfg["logging"]["log_path"],
    )

    # Pre-flight: consult the postmortem DB. Blockers abort here.
    pre = run_preflight("agent", live_money=args.live_money)
    print(pre.to_text())
    if not pre.passed:
        logger.error("Preflight blocked; aborting.")
        return 3

    # Always need a broker, even in dry-run, so we can read equity/positions.
    # Live money goes through the promotion gate via the broker constructor.
    executor = get_broker(args.broker, live_money=args.live_money)
    try:
        equity = executor.get_account_equity()
    except NotImplementedError:
        logger.warning("Broker %s isn't fully implemented; equity unavailable.",
                       args.broker)
        equity = 0.0
    endpoint = ("paper" if getattr(executor, "_paper", True) else "LIVE")
    logger.info("Agent broker connected — broker=%s endpoint=%s equity=$%.2f",
                args.broker, endpoint, equity)

    agent_cfg = AgentConfig(
        universe=cfg["data"]["universe"],
        seq_len=cfg["model"]["seq_len"],
        tick_seconds=args.tick_seconds,
        max_positions=args.max_positions,
        max_daily_loss_pct=args.max_daily_loss_pct,
        pre_close_minutes=args.pre_close_minutes,
        confidence_threshold=cfg["signals"]["confidence_threshold"],
        dry_run=args.dry_run,
        signal_dir=Path(cfg["signals"]["output_dir"]),
    )

    fetcher = _make_data_fetcher(cfg, args.source, args.score_sentiment)
    engineer = FeatureEngineer(window=cfg["model"]["seq_len"])
    ensemble = _build_ensemble()

    agent = TradingAgent(
        agent_cfg,
        executor=executor,
        data_fetcher=fetcher,
        feature_engineer=engineer,
        ensemble=ensemble,
        signal_builder=generate_signal,
        broker_name=args.broker,
    )
    agent.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
