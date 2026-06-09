"""Tests for the master-fix batch (fixes 1-9).

Run with:
    pytest tests/test_master_fixes.py -v
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ─────────────────────────────────────────────────────────────────

def _base_risk(**overrides):
    base = {
        "account_equity": 100_000.0,
        "entry_price": 500.0,
        "atr": 5.0,
        "vix": 13.0,
        "realized_vol": 0.01,
        "current_time": datetime(2026, 6, 9, 15, 0, tzinfo=timezone.utc),
        "existing_positions": [],
        "correlation_matrix": {},
    }
    base.update(overrides)
    return base


def _model(direction="long", confidence=0.85):
    return {
        "direction": direction,
        "confidence": confidence,
        "expected_return_pct": 0.03,
        "iv_change_pct": 0.0,
    }


# ── Fix 1: Regime side-bias blocks shorts in long_only ───────────────────────

class TestSideBiasFilter:
    def test_short_blocked_in_long_only(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(side_bias="long_only")
        result = generate_signal("SPY", _model("short", 0.90), risk)
        assert result is None, "Short should be suppressed in long_only regime"

    def test_long_passes_in_long_only(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(side_bias="long_only")
        result = generate_signal("SPY", _model("long", 0.90), risk)
        assert result is not None, "Long should pass in long_only regime"
        assert result["direction"] == "long"

    def test_both_directions_allowed_in_neutral(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(side_bias="both")
        # Long passes
        assert generate_signal("SPY", _model("long", 0.90), risk) is not None
        # Short also passes (regime allows it)
        assert generate_signal("SPY", _model("short", 0.90), risk) is not None

    def test_long_blocked_in_short_only(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(side_bias="short_only")
        result = generate_signal("SPY", _model("long", 0.90), risk)
        assert result is None, "Long should be suppressed in short_only regime"

    def test_regime_read_from_cache_when_not_in_risk_params(self, tmp_path, monkeypatch):
        """Without side_bias in risk_params, reads from regime_cache.json."""
        from src.signals import signal_generator as sg
        fake_cache = tmp_path / "regime_cache.json"
        fake_cache.write_text(json.dumps({"side_bias": "long_only"}))
        monkeypatch.setattr(sg, "_REGIME_CACHE", fake_cache)
        risk = _base_risk()  # no side_bias key
        result = sg.generate_signal("QQQ", _model("short", 0.90), risk)
        assert result is None


# ── Fix 2: Confidence threshold at 0.75 ─────────────────────────────────────

class TestConfidenceThreshold:
    def test_threshold_is_075(self):
        from src.signals.signal_generator import CONFIDENCE_THRESHOLD
        assert CONFIDENCE_THRESHOLD == 0.75, (
            f"Threshold should be 0.75, got {CONFIDENCE_THRESHOLD}"
        )

    def test_signal_below_075_suppressed(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(side_bias="both")
        result = generate_signal("SPY", _model("long", 0.72), risk)
        assert result is None, "0.72 confidence should be suppressed"

    def test_signal_at_075_passes(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(side_bias="both")
        result = generate_signal("SPY", _model("long", 0.75), risk)
        assert result is not None, "0.75 confidence should pass"

    def test_agent_config_default_is_075(self):
        from src.agent.trading_agent import AgentConfig
        import inspect
        sig = inspect.signature(AgentConfig)
        default = sig.parameters["confidence_threshold"].default
        assert default == 0.75, f"AgentConfig default should be 0.75, got {default}"


# ── Fix 4: RL agent weight > 0 ──────────────────────────────────────────────

class TestRLWeight:
    def test_rl_weight_nonzero(self):
        from src.model.ensemble import EnsembleWeights
        w = EnsembleWeights()
        assert w.rl > 0, f"RL weight should be > 0, got {w.rl}"

    def test_rl_normalised_weight_is_10pct(self):
        from src.model.ensemble import EnsembleWeights
        w = EnsembleWeights()
        nw = w.normalised()
        assert abs(nw["rl"] - 0.10) < 0.01, (
            f"Normalised RL weight should be ~10%, got {nw['rl']:.3f}"
        )

    def test_weights_sum_to_one(self):
        from src.model.ensemble import EnsembleWeights
        nw = EnsembleWeights().normalised()
        assert abs(sum(nw.values()) - 1.0) < 1e-9


# ── Fix 6: Calibrator bootstrap prior loads ─────────────────────────────────

class TestCalibratorBootstrap:
    def test_bootstrap_file_exists(self):
        assert (ROOT / "data" / "calibration_bootstrap.json").exists(), (
            "Run: python scripts/bootstrap_calibrator.py"
        )

    def test_above_threshold_bucket_scale_above_1(self):
        """High-confidence long buckets should be boosted, not penalised."""
        data = json.loads((ROOT / "data" / "calibration_bootstrap.json").read_text())
        cal = data["calibration"]
        above = {k: v for k, v in cal.items() if float(k) >= 0.75}
        assert above, "No above-threshold buckets found"
        for k, v in above.items():
            assert v > 1.0, (
                f"Bucket {k} scale={v} should be > 1.0 (long WR=66.7%)"
            )

    def test_calibrator_uses_bootstrap_when_no_live_trades(self, tmp_path, monkeypatch):
        """With empty journal, calibrator falls back to bootstrap prior."""
        from src.learning import signal_calibrator as sc
        # Point journal to empty file
        empty_journal = tmp_path / "journal.jsonl"
        empty_journal.write_text("")
        monkeypatch.setattr(sc, "JOURNAL_PATH", empty_journal)
        # Point bootstrap to the real file
        cal = sc.SignalCalibrator()
        # 0.80 is in the bootstrap range (0.75-0.85)
        raw = 0.80
        calibrated = cal.calibrate(raw)
        # With long WR=66.7%, calibrated should be slightly above raw
        assert calibrated >= raw, (
            f"Calibrated ({calibrated:.4f}) should be >= raw ({raw}) when long WR=66.7%"
        )


# ── Fix 7: Event blackout ────────────────────────────────────────────────────

class TestEventBlackout:
    def test_fomc_blocks_entry(self):
        from src.filters.event_blackout import is_blocked
        # FOMC June 18 2026 at 18:00 UTC — check at 17:30 (30 min before)
        fomc_minus_30 = datetime(2026, 6, 18, 17, 30, tzinfo=timezone.utc)
        blocked, reason = is_blocked("SPY", now=fomc_minus_30,
                                      skip_earnings_check=True)
        assert blocked, f"Should be blocked near FOMC, got reason: {reason}"
        assert "FOMC" in reason

    def test_clear_day_allows_entry(self):
        from src.filters.event_blackout import is_blocked
        # Random Tuesday away from any event
        clear = datetime(2026, 4, 14, 15, 0, tzinfo=timezone.utc)
        blocked, reason = is_blocked("SPY", now=clear, skip_earnings_check=True)
        assert not blocked, f"Should not be blocked on a clear day: {reason}"

    def test_etfs_skip_earnings_check(self):
        """ETFs should never be blocked by earnings (they don't have them)."""
        from src.filters.event_blackout import is_blocked
        clear = datetime(2026, 4, 14, 15, 0, tzinfo=timezone.utc)
        # Even without skip_earnings_check, ETFs should pass
        blocked, reason = is_blocked("SPY", now=clear)
        assert not blocked


# ── Fix 8: Inverse ETF semantic block ────────────────────────────────────────

class TestInverseETFBlock:
    def test_tqqq_blocked_when_sqqq_open(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(
            side_bias="both",
            existing_positions=[{"symbol": "SQQQ"}],
        )
        result = generate_signal("TQQQ", _model("long", 0.90), risk)
        assert result is None, "TQQQ should be blocked when SQQQ is open"

    def test_sqqq_blocked_when_tqqq_open(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(
            side_bias="both",
            existing_positions=[{"symbol": "TQQQ"}],
        )
        result = generate_signal("SQQQ", _model("short", 0.90), risk)
        assert result is None, "SQQQ should be blocked when TQQQ is open"

    def test_non_inverse_pair_not_blocked(self):
        """SPY and QQQ are not semantic inverses — should not block each other."""
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(
            side_bias="both",
            existing_positions=[{"symbol": "QQQ"}],  # dict form as the broker returns
        )
        result = generate_signal("SPY", _model("long", 0.90), risk)
        assert result is not None, "SPY should not be blocked by open QQQ"

    def test_inverse_check_case_insensitive(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(
            side_bias="both",
            existing_positions=[{"symbol": "sqqq"}],   # lowercase
        )
        result = generate_signal("TQQQ", _model("long", 0.90), risk)
        assert result is None


# ── Position size sanity (NaN regression) ────────────────────────────────────

class TestPositionSize:
    def test_valid_long_signal_has_nonnan_size(self):
        from src.signals.signal_generator import generate_signal
        risk = _base_risk(side_bias="both")
        result = generate_signal("NVDA", _model("long", 0.85), risk)
        assert result is not None
        assert not math.isnan(result["position_size_usd"]), "Position size must not be NaN"
        assert result["position_size_usd"] > 0, "Position size must be > 0"

    def test_tp_rr_is_2_to_1(self):
        """Default TP should be 2×risk (learned_params optimal_rr=2.0)."""
        from src.risk.risk_manager import apply_stop_loss, apply_take_profit
        entry = 500.0
        atr = 10.0
        stop = apply_stop_loss(entry, "long", atr)
        tp = apply_take_profit(entry, stop, "long")
        risk = entry - stop
        reward = tp - entry
        assert abs(reward / risk - 2.0) < 0.05, (
            f"R:R should be ~2.0, got {reward/risk:.2f}"
        )
