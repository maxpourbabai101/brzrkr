"""Tests for src.utils.archiver.BatchArchiver."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.archiver import BatchArchiver


def _seed_batch(scenario_dir: Path) -> None:
    """Create two scenarios × two symbols, each with trades + equity."""
    for scen in ("covid_crash", "rally_q1"):
        for sym in ("SPY", "QQQ"):
            d = scenario_dir / scen / sym
            d.mkdir(parents=True, exist_ok=True)
            # Trades file
            pd.DataFrame([
                {"entry": 100, "stop": 99, "exit": 102, "qty": 1,
                 "direction": "long", "pnl": 2.0},
                {"entry": 100, "stop": 99, "exit": 99,  "qty": 1,
                 "direction": "long", "pnl": -1.0},
            ]).to_csv(d / "trades.csv", index=False)
            # Equity curve
            pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
                "equity":    [100_000 + i * 50 for i in range(20)],
            }).to_csv(d / "equity_curve.csv", index=False)


@pytest.fixture
def archiver(tmp_path):
    return BatchArchiver(
        scenario_dir=tmp_path / "scenario_runs",
        archive_dir=tmp_path / "archive",
        master_trades=tmp_path / "all_trades.csv",
        summary_log=tmp_path / "summary.csv",
    )


def test_archive_aggregates_trades_and_summaries(archiver):
    _seed_batch(archiver.scenario_dir)
    stats = archiver.archive_current_batch(batch_id="test_batch_1")
    assert stats.scenarios_archived == 2
    assert stats.trades_appended == 8   # 2 trades × 2 syms × 2 scens
    assert archiver.master_trades.exists()
    assert archiver.summary_log.exists()
    assert stats.archive_path.exists()
    # Per-scenario dirs should be gone
    assert not (archiver.scenario_dir / "covid_crash").exists()


def test_archive_tarball_contains_data(archiver):
    _seed_batch(archiver.scenario_dir)
    stats = archiver.archive_current_batch(batch_id="t1")
    with tarfile.open(stats.archive_path, "r:gz") as tar:
        names = tar.getnames()
    # Should contain both scenarios + their CSVs
    assert any("covid_crash" in n for n in names)
    assert any("trades.csv" in n for n in names)


def test_archive_idempotent_when_empty(archiver):
    stats = archiver.archive_current_batch()
    assert stats.scenarios_archived == 0
    assert stats.trades_appended == 0


def test_two_batches_accumulate_in_master(archiver):
    _seed_batch(archiver.scenario_dir)
    archiver.archive_current_batch(batch_id="b1")
    _seed_batch(archiver.scenario_dir)
    archiver.archive_current_batch(batch_id="b2")
    master = pd.read_csv(archiver.master_trades)
    # 8 trades per batch × 2 batches = 16
    assert len(master) == 16
    assert set(master["batch_id"]) == {"b1", "b2"}


def test_trim_keeps_only_newest_n(archiver):
    archiver.keep_archives = 2
    for i in range(5):
        _seed_batch(archiver.scenario_dir)
        archiver.archive_current_batch(batch_id=f"b{i:02d}")
    archives = sorted(archiver.archive_dir.glob("batch_*.tar.gz"))
    assert len(archives) == 2


def test_stats_returns_expected_keys(archiver):
    _seed_batch(archiver.scenario_dir)
    archiver.archive_current_batch(batch_id="s1")
    s = archiver.stats()
    for key in ("master_trades_kb", "summary_log_kb",
                "archives_count", "archives_total_kb",
                "live_scenario_dir_kb"):
        assert key in s
    assert s["archives_count"] == 1
    assert s["master_trades_kb"] > 0


def test_optimizer_reads_aggregated_master(tmp_path, archiver):
    """End-to-end: archiver writes master_trades, optimizer picks it up."""
    _seed_batch(archiver.scenario_dir)
    archiver.archive_current_batch(batch_id="t1")
    from src.learning.profit_optimizer import ProfitOptimizer
    opt = ProfitOptimizer(
        trade_globs=[str(archiver.master_trades)],
        output_path=tmp_path / "params.json",
        min_trades=4,
    )
    result = opt.analyze()
    assert result.n_trades == 8
    assert result.win_rate == pytest.approx(0.5, abs=0.01)
