"""Autonomous trading loop.

Runs continuously during market hours, pulls fresh data, runs the
ensemble + risk pipeline, places bracket orders via Alpaca, and
applies daily/portfolio-level guardrails on top of the per-trade risk
manager.

The agent is designed to fail safe:

* SIGINT / SIGTERM → finish current tick, then exit cleanly.
* Existence of a sentinel file (``AGENT_STOP`` by default) → halt.
* Daily P&L breach → halt and stop trading for the session.
* Per-tick exception → log + continue (one bad tick should not bring
  down a 12-hour session).
* Market closed → log and sleep until next open (queried from broker
  clock, no hard-coded calendar).

This class is constructor-injected with every dependency it needs, so
tests can drop in stubs without monkey-patching imports.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.risk.countermeasures import CountermeasureSet
from src.execution.promotion_gate import PromotionGate
from src.learning.observer import SessionObservation, SessionObserver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    universe: Iterable[str]
    seq_len: int = 256
    tick_seconds: int = 300                  # 5 min between ticks
    max_positions: int = 5                   # cap on simultaneous open positions
    max_daily_loss_pct: float = 0.03         # halt at -3% daily drawdown
    pre_close_minutes: int = 15              # no new entries in last 15 min
    confidence_threshold: float = 0.29       # signal floor (regime detector updates this)
    dry_run: bool = False                    # log would-be trades, don't submit
    stop_file: Path = field(default_factory=lambda: Path("AGENT_STOP"))
    signal_dir: Path = field(default_factory=lambda: Path("data/signals"))
    heartbeat_every: int = 1                 # log portfolio every N ticks


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class TradingAgent:
    """Long-running loop wrapping data → features → predict → trade."""

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        executor,
        data_fetcher: Callable[[str], Dict[str, pd.DataFrame]],
        feature_engineer,
        ensemble,
        signal_builder: Callable[[str, Dict, Dict], Optional[Dict]],
        clock: Optional[Callable[[], Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        countermeasures: Optional[CountermeasureSet] = None,
        promotion_gate: Optional[PromotionGate] = None,
        broker_name: str = "alpaca",
    ) -> None:
        self.cfg = cfg
        self.executor = executor
        self.fetch_data = data_fetcher
        self.engineer = feature_engineer
        self.ensemble = ensemble
        self.build_signal = signal_builder
        # `clock` returns an Alpaca-like clock object with .is_open,
        # .timestamp, .next_open, .next_close attributes.
        self._clock = clock or (lambda: executor._client.get_clock())
        self._sleep = sleep

        # Risk countermeasures (stateful — circuit breaker, cooldowns, etc.)
        self.countermeasures = countermeasures or CountermeasureSet()
        self.promotion_gate = promotion_gate or PromotionGate()
        self.observer = SessionObserver()
        self.broker_name = broker_name

        # Mutable session state.
        self._stopped: bool = False
        self._sod_equity: Optional[float] = None
        self._sod_date: Optional[str] = None
        self._tick_count: int = 0
        self._trades_submitted: int = 0
        self._daily_breach: bool = False
        self._countermeasure_blocks: int = 0

        # SIGINT / SIGTERM → clean exit.
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                # Not in main thread — fine for tests.
                pass

        self.cfg.signal_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        logger.info(
            "TradingAgent starting — universe=%s dry_run=%s tick=%ds",
            list(self.cfg.universe), self.cfg.dry_run, self.cfg.tick_seconds,
        )
        logger.info(
            "To stop: Ctrl-C in this terminal, OR `touch %s` from another shell, "
            "OR click 'Stop agent' in the dashboard.",
            self.cfg.stop_file,
        )
        while not self._should_stop():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logger.exception("Tick failed — continuing")
            if self._should_stop():
                break
            self._interruptible_sleep(self._next_sleep_seconds())
        self._log_session_summary()

    # ------------------------------------------------------------------
    # Sleep helpers
    # ------------------------------------------------------------------
    def _next_sleep_seconds(self) -> float:
        """Default sleep is `tick_seconds`, but when the market is
        closed we extend it toward `next_open` (capped at 1 hour so
        we still poll the stop signals).
        """
        base = float(self.cfg.tick_seconds)
        try:
            clock = self._clock()
        except Exception:  # noqa: BLE001
            return base
        if getattr(clock, "is_open", True):
            return base

        next_open = getattr(clock, "next_open", None)
        ts = getattr(clock, "timestamp", None)
        if next_open is None:
            return base
        try:
            delta = (pd.Timestamp(next_open) - pd.Timestamp(ts or datetime.now(timezone.utc))).total_seconds()
        except Exception:  # noqa: BLE001
            return base
        # Sleep up to one hour when closed; pad by 5s so we wake just
        # after the bell instead of right before.
        return float(max(base, min(delta + 5.0, 3600.0)))

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in 2-second chunks so AGENT_STOP / SIGINT are caught
        within ~2 seconds regardless of tick_seconds."""
        if seconds <= 0:
            return
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._should_stop():
                return
            remaining = end - time.monotonic()
            self._sleep(min(2.0, remaining))

    def stop(self) -> None:
        self._stopped = True

    # ------------------------------------------------------------------
    # One tick = one model evaluation across the universe
    # ------------------------------------------------------------------
    def tick(self) -> None:
        self._tick_count += 1
        clock = self._clock()
        if not getattr(clock, "is_open", True):
            next_open = getattr(clock, "next_open", "unknown")
            logger.info("Market closed — sleeping until %s", next_open)
            return

        self._handle_day_rollover(clock)

        # Apply regime-aware parameter adjustments every 12 ticks (~1 hour).
        if self._tick_count % 12 == 1:
            try:
                from src.learning.regime_detector import current_regime
                regime = current_regime()
                # Override confidence threshold if the learner hasn't tuned it
                self.cfg.confidence_threshold = regime.confidence_threshold
                effective_max = max(
                    1,
                    int(self.cfg.max_positions * regime.max_positions_factor),
                )
                logger.info(
                    "Regime: %s (conf=%.2f) → conf_threshold=%.2f  "
                    "max_pos=%d  side_bias=%s",
                    regime.label, regime.confidence,
                    regime.confidence_threshold,
                    effective_max, regime.side_bias,
                )
                self._regime = regime
            except Exception as _rex:
                logger.debug("Regime detection skipped: %s", _rex)

        # Daily loss breaker.
        equity_now = self.executor.get_account_equity()
        if self._sod_equity:
            loss = (self._sod_equity - equity_now) / self._sod_equity
            if loss >= self.cfg.max_daily_loss_pct:
                logger.error(
                    "DAILY LOSS BREACH: %.2f%% (>= %.2f%%) — halting",
                    loss * 100, self.cfg.max_daily_loss_pct * 100,
                )
                self._daily_breach = True
                self._stopped = True
                return

        # Position cap.
        positions = self.executor.get_open_positions()
        held: set[str] = {p["symbol"] for p in positions}
        if len(positions) >= self.cfg.max_positions:
            logger.info(
                "At max positions (%d) — managing existing, no new entries",
                len(positions),
            )
            self._heartbeat(equity_now, len(positions))
            return

        # Pre-close cutoff.
        if self._minutes_to_close(clock) < self.cfg.pre_close_minutes:
            logger.info(
                "Within %d min of close — no new entries",
                self.cfg.pre_close_minutes,
            )
            self._heartbeat(equity_now, len(positions))
            return

        # Per-symbol evaluation.
        for symbol in self.cfg.universe:
            if symbol in held:
                logger.debug("Already long/short %s — skipping", symbol)
                continue
            self._evaluate_symbol(symbol, equity_now)

        self._heartbeat(equity_now, len(positions))

    # ------------------------------------------------------------------
    # Per-symbol evaluation
    # ------------------------------------------------------------------
    def _evaluate_symbol(self, symbol: str, account_equity: float) -> None:
        try:
            bundle = self.fetch_data(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Data fetch failed for %s: %s", symbol, exc)
            return

        prices = bundle.get("prices")
        if prices is None or prices.empty or len(prices) < self.cfg.seq_len:
            n = 0 if prices is None else len(prices)
            logger.debug("%s: insufficient history (%d bars)", symbol, n)
            return

        features = self.engineer.build_features(bundle)
        if features.empty:
            return

        try:
            prediction = self.ensemble.predict(features)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ensemble predict failed for %s: %s", symbol, exc)
            return

        if prediction.get("confidence", 0) < self.cfg.confidence_threshold:
            logger.info(
                "%s: confidence %.2f below threshold %.2f — no trade",
                symbol, prediction.get("confidence", 0),
                self.cfg.confidence_threshold,
            )
            return

        ctx = features.attrs.get("context", {})

        # Portfolio-level countermeasures (circuit breaker, cooldown,
        # sector cap, vol regime, etc.).
        allowed, reason = self.countermeasures.allow_new_entry(
            symbol=symbol,
            existing_positions=self.executor.get_open_positions(),
            vix=ctx.get("vix_level"),
        )
        if not allowed:
            logger.info("%s: countermeasure blocked entry — %s", symbol, reason)
            self._countermeasure_blocks += 1
            return
        entry = float(prices["close"].iloc[-1])
        atr = float((prices["high"] - prices["low"]).tail(14).mean())
        risk_params = {
            "account_equity": account_equity,
            "entry_price": entry,
            "atr": atr,
            "vix": float(ctx.get("vix_level") or 0.0),
            "realized_vol": float(prices["close"].pct_change().tail(20).std() or 0.0),
            "current_time": datetime.now(timezone.utc),
            "existing_positions": self.executor.get_open_positions(),
            "correlation_matrix": {},
        }

        signal_dict = self.build_signal(symbol, prediction, risk_params)
        if signal_dict is None:
            return

        # Vol-scaled sizing: shrink notional in elevated-vol regimes.
        adjusted = self.countermeasures.adjust_notional(
            signal_dict["position_size_usd"], vix=ctx.get("vix_level"),
        )
        if adjusted <= 0:
            logger.info("%s: vol regime drove notional to 0 — skipping", symbol)
            return
        signal_dict["position_size_usd"] = adjusted

        # Always write the JSON record, even in dry_run.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.cfg.signal_dir / f"{symbol}_{ts}.json"
        path.write_text(json.dumps(signal_dict, indent=2, default=str))

        if self.cfg.dry_run:
            logger.info("[DRY RUN] would submit %s %s @ %.2f",
                        symbol, signal_dict["direction"], entry)
            return

        result = self.executor.submit_signal(signal_dict)
        if result.submitted:
            self._trades_submitted += 1
            logger.info("Order submitted: %s id=%s", symbol, result.order_id)
            # Record the signal in the trade journal for live-trades tracking
            # and online learning.  Wrapped in try/except so a journal failure
            # never takes down the agent.
            try:
                from src.learning.trade_journal import TradeJournal
                TradeJournal().record_signal(signal_dict)
            except Exception as _jex:
                logger.debug("Trade journal record failed (non-fatal): %s", _jex)
        else:
            logger.warning("Order NOT submitted for %s: %s", symbol, result.reason)

        # After each live submission try an incremental model update in the
        # background — only fires if >= 5 new closed trades have accumulated.
        try:
            from src.learning.online_learner import get_learner
            import threading as _th
            _th.Thread(
                target=get_learner().maybe_update, daemon=True,
            ).start()
        except Exception as _lex:
            logger.debug("OnlineLearner update skipped: %s", _lex)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _handle_day_rollover(self, clock) -> None:
        # Pull the broker's view of "today" so timezone math stays consistent.
        ts = getattr(clock, "timestamp", None) or datetime.now(timezone.utc)
        date_str = pd.Timestamp(ts).date().isoformat()
        if date_str != self._sod_date:
            self._sod_date = date_str
            self._sod_equity = self.executor.get_account_equity()
            self._trades_submitted = 0
            logger.info(
                "New trading day %s — start-of-day equity $%.2f",
                date_str, self._sod_equity,
            )

    def _minutes_to_close(self, clock) -> float:
        nc = getattr(clock, "next_close", None)
        ts = getattr(clock, "timestamp", None) or datetime.now(timezone.utc)
        if nc is None:
            return float("inf")
        return (pd.Timestamp(nc) - pd.Timestamp(ts)).total_seconds() / 60.0

    def _heartbeat(self, equity: float, open_positions: int) -> None:
        if self._tick_count % self.cfg.heartbeat_every == 0:
            logger.info(
                "♥ tick=%d equity=$%.2f open_positions=%d submitted_today=%d",
                self._tick_count, equity, open_positions, self._trades_submitted,
            )

    def _should_stop(self) -> bool:
        if self._stopped:
            return True
        if self.cfg.stop_file.exists():
            logger.info("Stop file %s detected — halting", self.cfg.stop_file)
            return True
        return False

    def _on_signal(self, signum, _frame) -> None:
        logger.info("Signal %d received — stopping after current tick", signum)
        self._stopped = True

    def _log_session_summary(self) -> None:
        try:
            equity = self.executor.get_account_equity()
            positions = self.executor.get_open_positions()
        except Exception:  # noqa: BLE001
            equity, positions = -1.0, []
        sod = self._sod_equity or equity
        pnl = equity - sod
        pnl_pct = (pnl / sod * 100.0) if sod > 0 else 0.0
        logger.info(
            "=== Agent stopped === ticks=%d trades=%d open=%d "
            "equity=$%.2f sod=$%.2f session_pnl=$%+.2f (%.2f%%)",
            self._tick_count, self._trades_submitted, len(positions),
            equity, sod, pnl, pnl_pct,
        )

        # Feed the self-learning observer FIRST so we can attribute
        # lesson firings to this session in the track record.
        lessons_fired: list = []
        try:
            snap = SessionObservation(
                sod_equity=sod if sod > 0 else equity,
                end_equity=equity,
                trades_submitted=self._trades_submitted,
                ticks=self._tick_count,
                daily_breach=self._daily_breach,
                circuit_breaker_fired=(self.countermeasures._consecutive_losses
                                        >= self.countermeasures.cfg.consecutive_loss_limit),
                cooldown_fired=(self.countermeasures._last_loss_ts is not None),
                countermeasure_blocks=self._countermeasure_blocks,
            )
            result = self.observer.observe(snap)
            lessons_fired = list(set(result["confirmed"] + result["added"]))
            if lessons_fired:
                logger.info("Observer updated postmortem DB: confirmed=%s, added=%s",
                             result["confirmed"], result["added"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Observer failed: %s", exc)

        # Record session into the promotion gate's track record. This is
        # the data the gate reads to decide whether to allow live money,
        # AND the data the correlation analyzer reads to compute conditional
        # P&L per lesson.
        try:
            endpoint = "paper"
            paper_attr = getattr(self.executor, "_paper", None)
            if paper_attr is False:
                endpoint = "live"
            self.promotion_gate.record_session(
                broker=self.broker_name,
                endpoint=endpoint,
                start_equity=sod if sod > 0 else equity,
                end_equity=equity,
                trades_submitted=self._trades_submitted,
                ticks=self._tick_count,
                breach_triggered=self._daily_breach,
                notes=("dry_run" if self.cfg.dry_run else ""),
                lessons_fired=lessons_fired,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to record session in promotion gate: %s", exc)
