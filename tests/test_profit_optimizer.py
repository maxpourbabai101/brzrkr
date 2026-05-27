"""Tests for src.learning.profit_optimizer.ProfitOptimizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.learning.profit_optimizer import (
    OptimizationResult, ProfitOptimizer, load_learned_params,
)


def _write_trades(path: Path, trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(path, index=False)


@pytest.fixture
def empty_opt(tmp_path):
    return ProfitOptimizer(
        trade_globs=[str(tmp_path / "*.csv")],
        output_path=tmp_path / "params.json",
    )


def test_no_trades_returns_default(empty_opt):
    result = empty_opt.analyze()
    assert result.n_trades == 0
    assert "no trades.csv" in result.notes


def test_below_min_trades_falls_back(tmp_path):
    _write_trades(tmp_path / "trades.csv", [
        {"entry": 100, "stop": 99, "pnl": 1, "qty": 1, "direction": "long"},
    ] * 5)
    opt = ProfitOptimizer(
        trade_globs=[str(tmp_path / "*.csv")],
        output_path=tmp_path / "params.json",
        min_trades=20,
    )
    result = opt.analyze()
    assert result.n_trades == 5
    assert "need 20+" in result.notes


def test_computes_winning_stats(tmp_path):
    # 60 trades: 40 wins of +2R, 20 losses of -1R
    trades = []
    for _ in range(40):
        # Win: entry 100, stop 99, exit 102 → +2R
        trades.append({"entry": 100, "stop": 99, "exit": 102, "qty": 1,
                       "direction": "long", "pnl": 2.0})
    for _ in range(20):
        # Loss: entry 100, stop 99, exit 99 → -1R
        trades.append({"entry": 100, "stop": 99, "exit": 99, "qty": 1,
                       "direction": "long", "pnl": -1.0})
    _write_trades(tmp_path / "trades.csv", trades)

    opt = ProfitOptimizer(
        trade_globs=[str(tmp_path / "*.csv")],
        output_path=tmp_path / "params.json",
        min_trades=10,
    )
    result = opt.analyze()
    assert result.n_trades == 60
    assert result.win_rate == pytest.approx(40 / 60, abs=0.01)
    assert result.avg_win_R == pytest.approx(2.0, abs=0.01)
    assert result.avg_loss_R == pytest.approx(-1.0, abs=0.01)
    # EV: 40/60 * 2 + 20/60 * -1 = 1.333 - 0.333 = 1.0
    assert result.expected_R == pytest.approx(1.0, abs=0.01)
    # Profit factor: 40*2 / 20*1 = 4.0
    assert result.profit_factor == pytest.approx(4.0, abs=0.01)


def test_writes_json_when_apply(tmp_path):
    trades = [
        {"entry": 100, "stop": 99, "exit": 102, "qty": 1,
         "direction": "long", "pnl": 2.0}
    ] * 30
    _write_trades(tmp_path / "trades.csv", trades)
    opt = ProfitOptimizer(
        trade_globs=[str(tmp_path / "*.csv")],
        output_path=tmp_path / "params.json",
        min_trades=10,
    )
    result = opt.analyze()
    path = opt.write(result)
    assert path.exists()
    loaded = load_learned_params(path)
    assert loaded["n_trades"] == 30
    assert loaded["optimal_rr_ratio"] > 0


def test_grid_searches_rr(tmp_path):
    # Create asymmetric data — half hit big at +3R, half cap at -1R.
    trades = []
    for _ in range(30):
        trades.append({"entry": 100, "stop": 99, "exit": 103, "qty": 1,
                       "direction": "long", "pnl": 3.0})
    for _ in range(20):
        trades.append({"entry": 100, "stop": 99, "exit": 99, "qty": 1,
                       "direction": "long", "pnl": -1.0})
    _write_trades(tmp_path / "trades.csv", trades)
    opt = ProfitOptimizer(
        trade_globs=[str(tmp_path / "*.csv")],
        output_path=tmp_path / "params.json",
        min_trades=10,
    )
    result = opt.analyze()
    # Best RR should pick something near 3 (or wherever EV is maximised).
    assert result.optimal_rr_ratio in opt.rr_candidates
    assert result.expected_R > 0


def test_by_direction_split(tmp_path):
    trades = []
    for _ in range(15):
        trades.append({"entry": 100, "stop": 99, "exit": 102, "qty": 1,
                       "direction": "long", "pnl": 2.0})
    for _ in range(15):
        trades.append({"entry": 100, "stop": 101, "exit": 98, "qty": 1,
                       "direction": "short", "pnl": 2.0})
    _write_trades(tmp_path / "trades.csv", trades)
    opt = ProfitOptimizer(
        trade_globs=[str(tmp_path / "*.csv")],
        output_path=tmp_path / "params.json",
        min_trades=10,
    )
    result = opt.analyze()
    assert "long" in result.by_direction
    assert "short" in result.by_direction
    assert result.by_direction["long"]["n_trades"] == 15
    assert result.by_direction["short"]["n_trades"] == 15


def test_load_learned_params_missing_file(tmp_path):
    assert load_learned_params(tmp_path / "nope.json") is None


def test_load_learned_params_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json {")
    assert load_learned_params(p) is None
