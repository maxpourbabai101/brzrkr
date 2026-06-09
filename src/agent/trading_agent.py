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
    max_positions: int = 8                   # raised from 5 — more active, better diversification
    max_daily_loss_pct: float = 0.03         # halt at -3% daily drawdown
    pre_close_minutes: int = 15              # no new entries in last 15 min
    confidence_threshold: float = 0.75       # data: long WR=66.7% at 0.75; regime detector may adjust
    dry_run: bool = False                    # log would-be trades, don't submit
    stop_file: Path = field(default_factory=lambda: Path("AGENT_STOP"))
    signal_dir: Path = field(default_factory=lambda: Path("data/signals"))
    heartbeat_every: int = 1                 # log portfolio every N ticks
    replacement_min_confidence: float = 0.80 # raised proportionally with threshold
    replacement_multiplier: float = 1.50     # new signal must score 1.5× the weakest position


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

        # Six alpha scanners (lazy-import-safe singleton)
        self._composite_scanner = None   # initialised on first tick
        self._calibrator = None   # SignalCalibrator — fitted lazily on first tick
        self._weight_tracker = None  # DynamicWeightTracker — lazy init

        # Trailing stop manager — updates stops as positions move in our favour
        from src.execution.trailing_stop_manager import TrailingStopManager
        self._trailing_stop_mgr = TrailingStopManager()

        # ML drawdown predictor — auto-loaded if models/drawdown_model.json exists
        from src.risk.drawdown_predictor import DrawdownPredictor
        self._drawdown_predictor = DrawdownPredictor.load_if_exists()

        # Mutable session state.
        self._stopped: bool = False
        self._sod_equity: Optional[float] = None
        self._sod_date: Optional[str] = None
        self._tick_count: int = 0
        self._trades_submitted: int = 0
        self._daily_breach: bool = False
        self._countermeasure_blocks: int = 0

        # Latest regime result — default to neutral so side_bias never crashes.
        from src.learning.regime_detector import RegimeResult
        self._regime: RegimeResult = RegimeResult()  # safe default until first refresh

        # SIGINT / SIGTERM → clean exit.
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                # Not in main thread — fine for tests.
                pass

        self.cfg.signal_dir.mkdir(parents=True, exist_ok=True)

    def _get_scanner(self):
        if self._composite_scanner is None:
            try:
                from src.signals.advanced_scanners import CompositeScanner
                self._composite_scanner = CompositeScanner()
                logger.info("CompositeScanner (6 alpha scanners) initialised")
            except Exception as exc:
                logger.warning("CompositeScanner unavailable: %s", exc)
        return self._composite_scanner

    def _get_calibrator(self):
        if self._calibrator is None:
            try:
                from src.learning.signal_calibrator import get_calibrator
                self._calibrator = get_calibrator()
                logger.info("SignalCalibrator loaded (%d trades, %d buckets)",
                            self._calibrator._n_trades,
                            len(self._calibrator._calibration))
            except Exception as exc:
                logger.warning("SignalCalibrator unavailable: %s", exc)
        return self._calibrator

    def _get_weight_tracker(self):
        if self._weight_tracker is None:
            try:
                from src.model.dynamic_weights import get_weight_tracker
                self._weight_tracker = get_weight_tracker()
            except Exception as exc:
                logger.warning("DynamicWeightTracker unavailable: %s", exc)
        return self._weight_tracker

    # ------------------------------------------------------------------
    # Trade replacement helpers
    # ------------------------------------------------------------------
    def _score_position(self, broker_pos: dict, journal_entry: Optional[dict]) -> float:
        """Score how 'worth keeping' an open position is.

        Returns a float in roughly [0, 2]:
          > 1.0  — position is healthy (good remaining RR, in profit)
          < 0.5  — position is deteriorating (near stop, below entry, old)
          0.0    — no data / can't score

        Scoring components
        ------------------
        1. Remaining risk/reward: (tp - current) / (current - stop)  for longs
        2. P&L factor: slight bonus if in profit, penalty if deep underwater
        3. Position age: decays after 7 days — stale positions free slots
        """
        sym = broker_pos.get("symbol", "")
        prices = getattr(self, "_last_prices_map", {}).get(sym)
        if prices is None or prices.empty:
            return 0.5   # neutral — can't price, don't replace blindly

        current = float(prices["close"].iloc[-1])

        if journal_entry is None:
            return 0.4   # no journal context — lean toward replacing

        entry_price = float(journal_entry.get("entry_price", current))
        stop        = float(journal_entry.get("stop_price",  entry_price * 0.96))
        tp          = float(journal_entry.get("tp_price",    entry_price * 1.06))
        side        = journal_entry.get("side", "long")
        confidence  = float(journal_entry.get("confidence", 0.5))

        # ── Remaining risk/reward ──────────────────────────────────────
        if side == "long":
            remaining_upside = tp - current
            remaining_risk   = current - stop
        else:
            remaining_upside = current - tp
            remaining_risk   = stop - current

        if remaining_risk <= 0:
            rr_score = 2.0   # already past stop (broker should've filled) — score high
        elif remaining_upside <= 0:
            rr_score = 0.0   # trade has blown past TP (shouldn't happen) or reversing
        else:
            rr_score = remaining_upside / max(remaining_risk, 0.01)

        # ── P&L factor (never kick out a winner up 3%+) ────────────────
        pnl_pct = ((current - entry_price) / max(entry_price, 1e-6)
                   if side == "long"
                   else (entry_price - current) / max(entry_price, 1e-6))
        if pnl_pct >= 0.03:
            return 3.0   # winner — never replace

        pnl_factor = 1.0 + float(np.clip(pnl_pct * 5, -0.5, 0.3))

        # ── Age factor: decay score for positions open > 7 days ────────
        age_days = 0
        try:
            from datetime import datetime, timezone
            open_ts = journal_entry.get("open_ts", "")
            if open_ts:
                opened = datetime.fromisoformat(str(open_ts).replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - opened).days
        except Exception:
            pass
        age_factor = max(0.50, 1.0 - age_days * 0.05)  # -5 % score per day, floor 50 %

        score = rr_score * pnl_factor * age_factor * confidence
        return float(max(0.0, score))

    def _consider_trade_replacement(
        self,
        symbol: str,
        new_confidence: float,
        new_prediction: dict,
    ) -> bool:
        """Check if a new high-conviction signal should replace the weakest position.

        Called only when at max_positions.  Returns True if a slot was freed
        (a weak position was closed to make room for the new signal).

        Rules
        -----
        1. New signal confidence must be >= replacement_min_confidence (0.72).
        2. New signal opportunity score must be >= replacement_multiplier × worst score.
        3. Never replace a position that is up 3 %+ (let winners run).
        """
        # Gate: new signal must clear a high bar to justify displacing anything.
        if new_confidence < self.cfg.replacement_min_confidence:
            logger.debug(
                "%s: replacement skipped — confidence %.3f < %.3f",
                symbol, new_confidence, self.cfg.replacement_min_confidence,
            )
            return False

        try:
            positions = self.executor.get_open_positions()
        except Exception:
            return False

        if len(positions) < self.cfg.max_positions:
            return True   # slot already open, no replacement needed

        # Load the journal for stop/tp context on each existing position
        try:
            from src.learning.trade_journal import TradeJournal
            journal_map = {
                e["symbol"]: e
                for e in TradeJournal()._load_all()
                if e.get("status") == "open"
            }
        except Exception:
            journal_map = {}

        # Score all existing positions
        scored = []
        for pos in positions:
            s = self._score_position(pos, journal_map.get(pos.get("symbol", "")))
            scored.append((s, pos))
        scored.sort(key=lambda x: x[0])   # weakest first

        worst_score, worst_pos = scored[0]
        worst_sym = worst_pos.get("symbol", "")

        # New signal opportunity score
        expected_ret = abs(float(new_prediction.get("expected_return_pct", 0.01)))
        new_score = new_confidence * (1.0 + expected_ret * 8.0)

        if new_score < worst_score * self.cfg.replacement_multiplier:
            logger.info(
                "%s: no replacement — new_score=%.3f < %.1f × worst (%s, %.3f)",
                symbol, new_score, self.cfg.replacement_multiplier,
                worst_sym, worst_score,
            )
            return False

        # Execute: close the weakest position to free a slot
        logger.warning(
            "TRADE REPLACEMENT: closing %s (score=%.3f) to open %s (score=%.3f, conf=%.3f)",
            worst_sym, worst_score, symbol, new_score, new_confidence,
        )
        if self.cfg.dry_run:
            logger.info("[DRY RUN] would close %s for %s", worst_sym, symbol)
            return True   # pretend the slot was freed in dry-run

        closed = self.executor.close_position(worst_sym)
        if closed:
            # Update the journal immediately so position count is accurate
            try:
                from src.learning.trade_journal import TradeJournal
                prices = getattr(self, "_last_prices_map", {}).get(worst_sym)
                exit_price = float(prices["close"].iloc[-1]) if prices is not None else 0.0
                if exit_price > 0:
                    TradeJournal().close_trade(worst_sym, exit_price)
            except Exception as _je:
                logger.debug("Journal update on replacement failed: %s", _je)
            return True

        logger.warning("Trade replacement: close of %s failed — slot not freed", worst_sym)
        return False

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

        # Sync the journal against live broker state — mark any positions
        # that Alpaca closed (via TP/SL bracket legs) as "closed" in the
        # journal.  Must happen BEFORE the position-cap check so that freed
        # slots are immediately visible this tick.
        self._reconcile_journal()

        # Detect and protect any naked positions (no stop/tp orders).
        self._fix_naked_positions()

        # Update trailing stops for all open long positions
        try:
            from src.learning.trade_journal import TradeJournal
            journal_entries = TradeJournal()._load_all()
            # Build a minimal prices map from recent fetches (best-effort)
            self._trailing_stop_mgr.update_all(
                executor=self.executor,
                positions=self.executor.get_open_positions(),
                prices_map=getattr(self, "_last_prices_map", {}),
                journal_entries=journal_entries,
            )
        except Exception as _tse:
            logger.debug("Trailing stop update skipped: %s", _tse)

        # Apply regime-aware parameter adjustments every 12 ticks (~1 hour).
        if self._tick_count % 12 == 1:
            try:
                from src.learning.regime_detector import current_regime
                from src.learning.strategy_doctor import apply_doctor_adjustments

                regime = current_regime()
                self._regime = regime

                # Start with regime recommendations, then let the doctor
                # override with data-driven adjustments.
                self.cfg.confidence_threshold = regime.confidence_threshold
                effective_max = max(
                    1,
                    int(self.cfg.max_positions * regime.max_positions_factor),
                )

                # Doctor overlay: fine-tuned from actual closed-trade stats.
                doctor_cfg: Dict[str, Any] = apply_doctor_adjustments({
                    "confidence_threshold": self.cfg.confidence_threshold,
                })
                if "confidence_threshold" in doctor_cfg:
                    self.cfg.confidence_threshold = float(
                        doctor_cfg["confidence_threshold"]
                    )

                # Per-regime side_bias from doctor (data-driven) overrides
                # the regime default set in regime_detector.py.
                regime_side_bias = doctor_cfg.get("regime_side_bias", {})
                if regime.label in regime_side_bias:
                    regime.side_bias = regime_side_bias[regime.label]

                # Doctor's symbol blacklist
                self._symbol_blacklist: set = set(
                    doctor_cfg.get("symbol_blacklist", [])
                )

                logger.info(
                    "Regime: %s (conf=%.2f) → conf_threshold=%.2f  "
                    "max_pos=%d  side_bias=%s  blacklist=%s",
                    regime.label, regime.confidence,
                    self.cfg.confidence_threshold,
                    effective_max, regime.side_bias,
                    sorted(self._symbol_blacklist) or "none",
                )
            except Exception as _rex:
                logger.debug("Regime/doctor refresh skipped: %s", _rex)

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

        # Position cap — but keep evaluating even at max so trade replacement
        # can run for high-conviction new signals (replacement logic is inside
        # _evaluate_symbol_with_bundle when at cap).
        positions = self.executor.get_open_positions()
        held: set[str] = {p["symbol"] for p in positions}
        if len(positions) >= self.cfg.max_positions:
            logger.info(
                "At max positions (%d/%d) — evaluating for potential replacement",
                len(positions), self.cfg.max_positions,
            )

        # Pre-close cutoff.
        if self._minutes_to_close(clock) < self.cfg.pre_close_minutes:
            logger.info(
                "Within %d min of close — no new entries",
                self.cfg.pre_close_minutes,
            )
            self._heartbeat(equity_now, len(positions))
            return

        # Per-symbol evaluation — parallel data fetch, sequential signal eval.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        blacklist = getattr(self, "_symbol_blacklist", set())

        to_scan = [
            sym for sym in self.cfg.universe
            if sym not in held and sym not in blacklist
        ]

        # Fetch all symbol data in parallel (I/O-bound; 8 workers is safe for yfinance)
        bundles: dict = {}
        if to_scan:
            def _safe_fetch(sym):
                try:
                    return sym, self.fetch_data(sym)
                except Exception as exc:
                    logger.warning("Data fetch failed for %s: %s", sym, exc)
                    return sym, None

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_safe_fetch, sym): sym for sym in to_scan}
                for fut in as_completed(futures):
                    sym, bundle = fut.result()
                    if bundle is not None:
                        bundles[sym] = bundle

        # Sequential signal evaluation (ensemble + scanners are CPU-bound / single-threaded safe)
        for symbol in to_scan:
            bundle = bundles.get(symbol)
            if bundle is None:
                continue
            self._evaluate_symbol_with_bundle(symbol, equity_now, bundle)

        self._heartbeat(equity_now, len(positions))

    # ------------------------------------------------------------------
    # Per-symbol evaluation
    # ------------------------------------------------------------------
    def _evaluate_symbol_with_bundle(self, symbol: str, account_equity: float,
                                     bundle: dict) -> None:
        prices = bundle.get("prices")
        if prices is None or prices.empty or len(prices) < self.cfg.seq_len:
            n = 0 if prices is None else len(prices)
            logger.debug("%s: insufficient history (%d bars)", symbol, n)
            return

        features = self.engineer.build_features(bundle)
        if features.empty:
            return

        # Apply cross-sectional Z-score normalisation (fixes XGBoost feature scale issues)
        try:
            from src.features.normalizer import FeatureNormaliser
            features = FeatureNormaliser().transform(features)
        except Exception as _ne:
            logger.debug("Feature normalisation skipped: %s", _ne)

        # Cache prices for trailing stop manager
        if not hasattr(self, "_last_prices_map"):
            self._last_prices_map = {}
        prices_df = bundle.get("prices")
        if prices_df is not None and not prices_df.empty:
            self._last_prices_map[symbol] = prices_df

        # Refresh ensemble weights from dynamic tracker (shifts weight toward
        # whichever sub-model has been most accurate in the last 60 trades).
        try:
            tracker = self._get_weight_tracker()
            if tracker is not None:
                dyn = tracker.get_weights()
                self.ensemble.weights.lstm        = dyn["lstm"]
                self.ensemble.weights.xgboost     = dyn["xgboost"]
                self.ensemble.weights.transformer = dyn["transformer"]
        except Exception as _we:
            logger.debug("Dynamic weight refresh skipped: %s", _we)

        try:
            prediction = self.ensemble.predict(features)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ensemble predict failed for %s: %s", symbol, exc)
            return

        raw_confidence = float(prediction.get("confidence", 0))

        # Apply signal calibration (adjusts confidence based on actual win rates)
        calibrator = self._get_calibrator()
        if calibrator is not None:
            raw_confidence = calibrator.calibrate(raw_confidence)

        # Run the 6 alpha scanners — they can boost or penalise confidence
        # and must agree (or at least not oppose) the ensemble direction.
        scanner_boost  = 0.0
        scanner_dir    = "neutral"
        _scan_results  = []
        scanner        = self._get_scanner()
        if scanner is not None:
            try:
                scanner_boost, scanner_dir, _scan_results = scanner.run(symbol, prices)
            except Exception as _se:
                logger.debug("CompositeScanner failed for %s: %s", symbol, _se)

        # If scanners strongly oppose the ensemble direction, treat it as a veto.
        ensemble_dir = prediction.get("direction", "long")
        if scanner_dir not in ("neutral", ensemble_dir) and scanner_boost < -0.10:
            logger.info(
                "%s: scanner veto — ensemble=%s scanner=%s boost=%.3f",
                symbol, ensemble_dir, scanner_dir, scanner_boost,
            )
            return

        # Apply scanner boost to confidence
        adjusted_confidence = float(np.clip(raw_confidence + scanner_boost, 0.0, 1.0))

        # ── Trend bypass: if 5+ scanners agree AND it's a clear trend,
        # lower the effective threshold by 0.05 so grinding bull markets
        # don't get filtered out entirely.
        n_agreeing = sum(
            1 for r in _scan_results
            if r.direction == scanner_dir
        ) if _scan_results else 0

        effective_threshold = self.cfg.confidence_threshold
        if n_agreeing >= 5 and scanner_dir == ensemble_dir:
            effective_threshold = max(effective_threshold - 0.05, 0.40)
            logger.debug(
                "%s: 5+ scanners agree — threshold lowered %.2f→%.2f",
                symbol, self.cfg.confidence_threshold, effective_threshold,
            )

        if adjusted_confidence < effective_threshold:
            logger.info(
                "%s: confidence %.2f (raw=%.2f + scan=%.3f) below threshold %.2f — no trade",
                symbol, adjusted_confidence, raw_confidence, scanner_boost,
                effective_threshold,
            )
            return

        # Update prediction confidence with the scanner-adjusted value
        prediction = dict(prediction)
        prediction["confidence"] = adjusted_confidence

        # Regime side-bias filter — never fight the macro trend.
        direction = prediction.get("direction", "long")
        side_bias = getattr(self._regime, "side_bias", "both")
        if side_bias == "long_only" and direction == "short":
            logger.info(
                "%s: regime=%s side_bias=long_only — skipping short signal",
                symbol, getattr(self._regime, "label", "?"),
            )
            return
        if side_bias == "short_only" and direction == "long":
            logger.info(
                "%s: regime=%s side_bias=short_only — skipping long signal",
                symbol, getattr(self._regime, "label", "?"),
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

        # ── Trade replacement: if at cap, consider displacing the weakest position ─
        try:
            live_positions = self.executor.get_open_positions()
            if len(live_positions) >= self.cfg.max_positions:
                slot_freed = self._consider_trade_replacement(
                    symbol, adjusted_confidence, prediction,
                )
                if not slot_freed:
                    return   # still at cap and no replacement — skip
        except Exception as _rpe:
            logger.debug("Trade replacement check failed (non-fatal): %s", _rpe)

        # Vol-scaled sizing: shrink notional in elevated-vol regimes.
        adjusted = self.countermeasures.adjust_notional(
            signal_dict["position_size_usd"], vix=ctx.get("vix_level"),
        )
        if adjusted <= 0:
            logger.info("%s: vol regime drove notional to 0 — skipping", symbol)
            return
        signal_dict["position_size_usd"] = adjusted

        # ML drawdown sizing: further shrink notional when drawdown probability is high.
        if self._drawdown_predictor is not None:
            try:
                dd_prob = self._drawdown_predictor.predict_proba(features)
                dd_mult = self._drawdown_predictor.size_multiplier(dd_prob)
                if dd_mult < 1.0:
                    logger.info(
                        "%s: drawdown predictor prob=%.2f → size multiplier=%.2f",
                        symbol, dd_prob, dd_mult,
                    )
                signal_dict["position_size_usd"] *= dd_mult
                signal_dict["drawdown_prob"] = round(dd_prob, 4)
                signal_dict["drawdown_size_mult"] = round(dd_mult, 4)
            except Exception as _dd_err:
                logger.debug("Drawdown predictor failed (non-fatal): %s", _dd_err)

        # Always write the JSON record, even in dry_run.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.cfg.signal_dir / f"{symbol}_{ts}.json"

        # Record which scanners fired for attribution tracking
        if _scan_results:
            signal_dict["scanners_fired"] = [r.scanner for r in _scan_results
                                              if r.direction == ensemble_dir]

        # Store per-model directional predictions so _reconcile_journal can
        # later update the DynamicWeightTracker once the trade closes.
        components = prediction.get("components", {})
        if components:
            signal_dict["model_directions"] = {
                name: sub.get("direction", "long")
                for name, sub in components.items()
            }

        path.write_text(json.dumps(signal_dict, indent=2, default=str))
        try:
            from src.data.db import write_signal
            write_signal(signal_dict)
        except Exception as _db_err:
            logger.debug("DB signal write skipped: %s", _db_err)

        if self.cfg.dry_run:
            logger.info("[DRY RUN] would submit %s %s @ %.2f",
                        symbol, signal_dict["direction"], entry)
            return

        # Stamp the current regime into the signal so the doctor can segment.
        signal_dict["regime_label"] = getattr(self._regime, "label", "unknown")

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

        # After each live submission run incremental learning + doctor in bg.
        try:
            from src.learning.online_learner import get_learner
            from src.learning.strategy_doctor import get_doctor
            import threading as _th
            _th.Thread(target=get_learner().maybe_update, daemon=True).start()
            _th.Thread(target=get_doctor().maybe_run, daemon=True).start()
        except Exception as _lex:
            logger.debug("OnlineLearner/Doctor update skipped: %s", _lex)

    def _fix_naked_positions(self) -> None:
        """Detect positions with no stop-loss/take-profit orders and protect them.

        A position is 'naked' if there are no open STOP or LIMIT sell orders
        for that symbol.  This happens when:
        - The bracket was cancelled (seen with SPY)
        - The position was opened outside the agent
        - A partial fill orphaned the bracket legs

        Fix: read the journal for the stored stop/tp prices and place GTC orders.
        """
        try:
            broker_positions = self.executor.get_open_positions()
            if not broker_positions:
                return

            open_orders = self.executor.get_orders(status="open", limit=200)
            # Which symbols already have a protective order?
            protected_syms: set = set()
            for o in open_orders:
                otype = str(o.get("order_type", "")).lower()
                if "stop" in otype or "limit" in otype:
                    protected_syms.add(o.get("symbol"))

            naked = [p for p in broker_positions if p["symbol"] not in protected_syms]
            if not naked:
                return

            from src.learning.trade_journal import TradeJournal
            journal = TradeJournal()
            all_entries = {r["symbol"]: r for r in journal._load_all()
                           if r.get("status") == "open"}

            for pos in naked:
                sym   = pos["symbol"]
                qty   = abs(int(float(pos["qty"])))
                direc = "long" if float(pos.get("qty", 1)) > 0 else "short"

                entry = all_entries.get(sym)
                if entry:
                    sl  = float(entry.get("stop_loss", 0))
                    tp  = float(entry.get("take_profit", 0))
                else:
                    # Fallback: compute from current price using default ATR%
                    ep  = float(pos.get("avg_entry_price", pos.get("current_price", 0)))
                    sl  = round(ep * (0.967 if direc == "long" else 1.033), 2)  # ~3.3% stop
                    tp  = round(ep * (1.050 if direc == "long" else 0.950), 2)  # ~5% target

                if sl <= 0 or tp <= 0 or qty < 1:
                    logger.warning("_fix_naked_positions: bad params for %s — skipping", sym)
                    continue

                logger.warning(
                    "NAKED POSITION detected: %s qty=%d — placing GTC stop=$%.2f tp=$%.2f",
                    sym, qty, sl, tp,
                )
                self.executor.protect_position(sym, qty, sl, tp, direction=direc)

        except Exception as exc:
            logger.debug("_fix_naked_positions skipped (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Journal reconciliation — keeps journal in sync with broker state
    # ------------------------------------------------------------------
    def _reconcile_journal(self) -> None:
        """Compare open journal entries against live Alpaca positions and
        close out any entry whose position has already been exited at the
        broker (TP or SL bracket fill, manual close, or liquidation).

        Called at the top of every tick so the position-cap check that
        follows sees the correct count of truly open positions.
        """
        try:
            from src.learning.trade_journal import TradeJournal
            journal = TradeJournal()

            open_entries = [r for r in journal._load_all()
                            if r.get("status") == "open"]
            if not open_entries:
                return  # nothing to reconcile

            # Symbols currently held at the broker.
            held_symbols: set = {p["symbol"]
                                  for p in self.executor.get_open_positions()}

            # Build a map of recent closed orders: symbol → best fill info.
            # Prefer filled_avg_price; fall back to limit_price / stop_price
            # (those come from bracket legs).
            try:
                closed_orders = self.executor.get_orders(status="closed", limit=200)
            except Exception:
                closed_orders = []

            # Walk oldest-to-newest so the latest fill wins in the map.
            exit_map: Dict[str, Dict] = {}
            for o in sorted(closed_orders,
                             key=lambda x: x.get("filled_at") or ""):
                sym = o.get("symbol")
                price = (o.get("filled_avg_price")
                         or o.get("limit_price")
                         or o.get("stop_price"))
                if sym and price:
                    exit_map[sym] = {
                        "price": float(price),
                        "ts":    o.get("filled_at"),
                    }

            # Reconcile each open journal entry.
            reconciled = 0
            for entry in open_entries:
                sym = entry.get("symbol")
                if sym in held_symbols:
                    continue  # still open at broker — no action needed

                # Position is gone at the broker.
                info = exit_map.get(sym)
                if info:
                    closed = journal.close_trade(
                        sym,
                        info["price"],
                        exit_time=info.get("ts"),
                    )
                    if closed:
                        reconciled += 1
                        logger.info(
                            "Reconciled: %s closed at $%.2f (was open in journal)",
                            sym, info["price"],
                        )
                        # Record scanner attribution for this closed trade
                        try:
                            from src.learning.scanner_attribution import get_attributor
                            from src.signals.advanced_scanners import CompositeScanner
                            sig_file = list(Path("data/signals").glob(f"{sym}_*.json"))
                            if sig_file:
                                sig = json.loads(sig_file[-1].read_text())
                                scanners_fired = sig.get("scanners_fired", [])
                                all_scanner_names = [
                                    sc.__class__.__name__
                                    for sc in CompositeScanner().scanners
                                ]
                                closed_rec = journal._load_all()
                                this_trade = next((r for r in closed_rec
                                                   if r.get("symbol") == sym
                                                   and r.get("status") == "closed"), None)
                                if this_trade and scanners_fired:
                                    pnl = float(this_trade.get("pnl_pct", 0))
                                    get_attributor().record_trade(
                                        scanners_fired=scanners_fired,
                                        all_scanners=all_scanner_names,
                                        won=pnl > 0,
                                    )
                        except Exception as _ae:
                            logger.debug("Scanner attribution recording skipped: %s", _ae)

                        # Update per-model accuracy so DynamicWeightTracker can
                        # shift ensemble weights toward whichever sub-model has
                        # been most correct recently.
                        try:
                            from src.model.dynamic_weights import get_weight_tracker
                            sig_files = list(Path("data/signals").glob(f"{sym}_*.json"))
                            if sig_files:
                                sig_data = json.loads(sig_files[-1].read_text())
                                model_dirs = sig_data.get("model_directions", {})
                                if model_dirs:
                                    # Determine what the "correct" direction was.
                                    # Won & long → long was correct; Lost & long → short was.
                                    closed_records = journal._load_all()
                                    trade_rec = next(
                                        (r for r in closed_records
                                         if r.get("symbol") == sym
                                         and r.get("status") == "closed"),
                                        None,
                                    )
                                    if trade_rec:
                                        pnl = float(trade_rec.get("pnl_pct", 0))
                                        original_dir = sig_data.get("direction", "long")
                                        won = pnl > 0
                                        if original_dir == "long":
                                            actual_dir = "long" if won else "short"
                                        else:
                                            actual_dir = "short" if won else "long"
                                        wt = get_weight_tracker()
                                        for model_name, model_dir in model_dirs.items():
                                            wt.record(model_name, model_dir, actual_dir)
                                        logger.debug(
                                            "Dynamic weights updated: %s  won=%s  actual=%s",
                                            sym, won, actual_dir,
                                        )
                        except Exception as _we:
                            logger.debug("Dynamic weight update skipped: %s", _we)
                else:
                    # Broker has no fill info (race condition or very old
                    # entry).  Use the last known price from Alpaca's
                    # account data as a best-effort exit.
                    try:
                        current_price = self.executor.get_account_equity()  # fallback
                        # Try to get the symbol's last price from open orders.
                        all_orders = self.executor.get_orders(status="all", limit=50)
                        sym_orders = [x for x in all_orders if x.get("symbol") == sym]
                        if sym_orders:
                            p = (sym_orders[-1].get("filled_avg_price")
                                 or sym_orders[-1].get("limit_price")
                                 or sym_orders[-1].get("stop_price"))
                            if p:
                                journal.close_trade(sym, float(p))
                                reconciled += 1
                                logger.warning(
                                    "Reconciled %s (no fill_at): used fallback price $%.2f",
                                    sym, float(p),
                                )
                    except Exception as _fallback_exc:
                        logger.debug("Reconcile fallback failed for %s: %s",
                                     sym, _fallback_exc)

            if reconciled:
                logger.info(
                    "Journal reconciliation: %d position(s) marked closed",
                    reconciled,
                )

        except Exception as exc:
            logger.debug("_reconcile_journal skipped (non-fatal): %s", exc)

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
            # Run the data janitor in a background thread — once per day.
            # It cleans old signals, market vectors, stale scenario artifacts,
            # and excess archives without blocking the agent tick.
            import threading as _th
            def _run_janitor():
                try:
                    from src.maintenance.data_janitor import DataJanitor
                    report = DataJanitor().run()
                    if report.files_deleted or report.errors:
                        logger.info("DataJanitor: %s", report)
                except Exception as _je:
                    logger.debug("DataJanitor skipped: %s", _je)
            _th.Thread(target=_run_janitor, daemon=True, name="data-janitor").start()

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
        # Run the strategy doctor at end-of-session so tomorrow starts
        # with up-to-date calibrated parameters.
        try:
            from src.learning.strategy_doctor import get_doctor
            get_doctor().end_of_day()
            logger.info("StrategyDoctor EOD diagnosis complete — see data/doctor_report.md")
        except Exception as _de:
            logger.debug("StrategyDoctor EOD skipped: %s", _de)

        # Refit calibrator with session's new trades
        try:
            from src.learning.signal_calibrator import get_calibrator
            n = get_calibrator().fit()
            logger.info("SignalCalibrator refit on %d closed trades", n)
        except Exception:
            pass

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
