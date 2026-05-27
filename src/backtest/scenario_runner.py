"""ScenarioRunner — replay the ensemble through every historical episode.

For each scenario × symbol pair:
1. Fetch OHLCV for the exact date range (from Yahoo via the scraper).
2. Run :class:`BacktestRunner` with a predict function that feeds the
   data through :class:`FeatureEngineer` then the ensemble.
3. Record per-run metrics: P&L, trades, win rate, Sharpe, max DD,
   plus the buy-and-hold benchmark for context.
4. Aggregate into a single DataFrame and render a categorized report.

Designed to run unattended over a weekend. Each scenario writes its
own artifacts to ``data/scenario_runs/<scenario>/<symbol>/`` so a
crash midway through doesn't lose completed work.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.backtest.backtest_runner import BacktestRunner
from src.backtest.live_status import LiveStatusWriter
from src.backtest.scenarios import MarketScenario, all_scenarios

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    scenario: str
    category: str
    symbol: str
    start: str
    end: str
    bars: int
    trades: int
    win_rate: float
    avg_pnl: float
    sharpe: float
    max_drawdown_pct: float
    final_equity: float
    benchmark_return_pct: float
    relative_vs_benchmark_pct: float
    difficulty: str
    failed: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioRunner:
    ensemble_factory: Callable[[], Any]   # callable returning a fresh ensemble
    feature_engineer_factory: Optional[Callable[[], Any]] = None
    initial_equity: float = 100_000.0
    confidence_threshold: float = 0.75
    output_dir: Path = field(default_factory=lambda: Path("data/scenario_runs"))
    seq_len: int = 256
    live_writer: Optional[LiveStatusWriter] = None
    parallel_workers: int = 1            # if > 1, runs scenarios concurrently
    live_writers: List[LiveStatusWriter] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.feature_engineer_factory is None:
            from src.features.feature_engineer import FeatureEngineer
            self.feature_engineer_factory = lambda: FeatureEngineer(
                window=self.seq_len
            )
        # Build N live writers — one per slot. Even in sequential mode we
        # have slot 0 for backward compat.
        if not self.live_writers:
            n = max(1, self.parallel_workers)
            self.live_writers = [
                LiveStatusWriter(
                    path=self.output_dir / (
                        "_live.json" if i == 0 else f"_live_{i}.json"
                    )
                )
                for i in range(n)
            ]
        # Backward-compat single accessor
        if self.live_writer is None:
            self.live_writer = self.live_writers[0]
        self._scenarios_done = 0
        self._scenarios_done_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _fetch(self, symbol: str, scenario: MarketScenario) -> pd.DataFrame:
        """Pull OHLCV for a specific window. Tries Alpaca first if
        creds are present (higher rate limit), falls back to Yahoo
        scraper. Each call also has a hard timeout."""
        import os
        # Try Alpaca first — much better rate limits than Yahoo.
        if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"):
            try:
                from src.data_loader import fetch_futures
                df = fetch_futures(symbol, scenario.start, scenario.end)
                if not df.empty and len(df) > 5:
                    return df
            except Exception as exc:  # noqa: BLE001
                logger.debug("Alpaca fetch failed (%s/%s): %s",
                              scenario.name, symbol, exc)
        # Fall back to Yahoo via scraper.
        from src.data_scraper import WebDataScraper
        agent = WebDataScraper()
        return agent.scrape_ohlcv(
            symbol,
            start=scenario.start,
            end=scenario.end,
        )

    def _make_predict_fn(self, ensemble, engineer) -> Callable:
        """Wrap the ensemble so the backtester (which passes raw price
        windows) gets features-with-context underneath."""
        def predict(price_window: pd.DataFrame) -> Dict[str, Any]:
            try:
                bundle = {"prices": price_window}
                feats = engineer.build_features(bundle)
                if feats.empty:
                    return {"direction": "long",
                            "expected_return_pct": 0.0,
                            "iv_change_pct": 0.0,
                            "confidence": 0.0}
                return ensemble.predict(feats)
            except Exception as exc:  # noqa: BLE001
                logger.debug("predict_fn error: %s", exc)
                return {"direction": "long",
                        "expected_return_pct": 0.0,
                        "iv_change_pct": 0.0,
                        "confidence": 0.0}
        return predict

    # ------------------------------------------------------------------
    def run_one(
        self,
        scenario: MarketScenario,
        symbol: str,
        *,
        slot: int = 0,
    ) -> ScenarioResult:
        """Run a single scenario × symbol, streaming live status to the
        writer at index `slot`."""
        writer = self.live_writers[slot] if slot < len(self.live_writers) \
            else self.live_writer
        out_dir = self.output_dir / scenario.name / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            df = self._fetch(symbol, scenario)
            if df is None or df.empty:
                return ScenarioResult(
                    scenario=scenario.name, category=scenario.category,
                    symbol=symbol, start=str(scenario.start.date()),
                    end=str(scenario.end.date()), bars=0, trades=0,
                    win_rate=0.0, avg_pnl=0.0, sharpe=0.0,
                    max_drawdown_pct=0.0, final_equity=self.initial_equity,
                    benchmark_return_pct=0.0, relative_vs_benchmark_pct=0.0,
                    difficulty=scenario.expected_difficulty,
                    failed=True, error="no data",
                )

            # Window for features. If the scenario is shorter than seq_len,
            # backtest will skip — pad by pulling additional history.
            if len(df) <= self.seq_len:
                from src.data_scraper import WebDataScraper
                # Pull extra leading data.
                from datetime import timedelta
                padded_start = scenario.start - timedelta(days=self.seq_len + 30)
                df = WebDataScraper().scrape_ohlcv(
                    symbol, start=padded_start, end=scenario.end,
                )

            ensemble = self.ensemble_factory()
            engineer = self.feature_engineer_factory()
            predict_fn = self._make_predict_fn(ensemble, engineer)

            runner = BacktestRunner(
                predict_fn=predict_fn,
                initial_equity=self.initial_equity,
                confidence_threshold=self.confidence_threshold,
                output_dir=out_dir,
                live_writer=writer,
            )
            window = min(self.seq_len, max(64, len(df) // 4))

            # Tell the live writer we're starting this scenario.
            if writer is not None:
                bars_total = max(0, len(df) - window)
                writer.start(
                    scenario=scenario.name,
                    symbol=symbol,
                    category=scenario.category,
                    bars_total=bars_total,
                    initial_equity=self.initial_equity,
                )

            res = runner.run(df, window=window)

            if writer is not None:
                writer.finish()

            # Benchmark = buy-and-hold over the scenario window.
            try:
                # Filter to the scenario period for the benchmark calc.
                in_window = df[(df.index >= scenario.start) & (df.index <= scenario.end)]
                if len(in_window) >= 2:
                    benchmark = ((in_window["close"].iloc[-1]
                                  / in_window["close"].iloc[0]) - 1) * 100.0
                else:
                    benchmark = 0.0
            except Exception:
                benchmark = 0.0

            strat_return = ((res.summary["final_equity"] - self.initial_equity)
                            / self.initial_equity * 100.0)
            relative = strat_return - benchmark

            return ScenarioResult(
                scenario=scenario.name, category=scenario.category,
                symbol=symbol, start=str(scenario.start.date()),
                end=str(scenario.end.date()), bars=len(df),
                trades=int(res.summary["trades"]),
                win_rate=float(res.summary["win_rate"]),
                avg_pnl=float(res.summary["avg_pnl"]),
                sharpe=float(res.summary["sharpe"]),
                max_drawdown_pct=float(res.summary["max_drawdown_pct"]),
                final_equity=float(res.summary["final_equity"]),
                benchmark_return_pct=benchmark,
                relative_vs_benchmark_pct=relative,
                difficulty=scenario.expected_difficulty,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error("scenario %s / %s failed: %s",
                         scenario.name, symbol, exc)
            traceback.print_exc()
            return ScenarioResult(
                scenario=scenario.name, category=scenario.category,
                symbol=symbol, start=str(scenario.start.date()),
                end=str(scenario.end.date()), bars=0, trades=0,
                win_rate=0.0, avg_pnl=0.0, sharpe=0.0,
                max_drawdown_pct=0.0, final_equity=self.initial_equity,
                benchmark_return_pct=0.0, relative_vs_benchmark_pct=0.0,
                difficulty=scenario.expected_difficulty,
                failed=True, error=str(exc)[:200],
            )

    def run_all(
        self,
        scenarios: Optional[List[MarketScenario]] = None,
        *,
        symbol_limit_per_scenario: Optional[int] = None,
    ) -> List[ScenarioResult]:
        """Run every scenario × symbol. Concurrent if parallel_workers > 1."""
        scenarios = scenarios or all_scenarios()

        # Build the flat work queue: list of (scenario, symbol).
        work: List[tuple] = []
        for sc in scenarios:
            syms = sc.symbols
            if symbol_limit_per_scenario:
                syms = syms[:symbol_limit_per_scenario]
            for sym in syms:
                work.append((sc, sym))
        total = len(work)

        # Broadcast overall progress on every writer (so any slot reports it).
        for w in self.live_writers:
            w.set_overall_progress(0, total)

        results: List[ScenarioResult] = []
        if self.parallel_workers <= 1:
            return self._run_sequential(work, total, results)
        return self._run_parallel(work, total, results)

    def _run_sequential(self, work, total, results):
        for i, (sc, sym) in enumerate(work, 1):
            for w in self.live_writers:
                w.set_overall_progress(i - 1, total)
            logger.info("[%d/%d] %s × %s …", i, total, sc.name, sym)
            t0 = time.time()
            res = self.run_one(sc, sym, slot=0)
            results.append(res)
            logger.info(
                "    finished in %.1fs  trades=%d  pnl=%+.2f%%  vs bench %+.2f%%",
                time.time() - t0, res.trades,
                (res.final_equity - self.initial_equity) / self.initial_equity * 100.0,
                res.relative_vs_benchmark_pct,
            )
            self._persist_partial(results)
        for w in self.live_writers:
            w.shutdown()
        return results

    def _run_parallel(self, work, total, results):
        """Dispatch work across N threads, each writing to its own slot."""
        n_workers = self.parallel_workers
        logger.info("Running %d scenarios across %d parallel workers",
                     total, n_workers)
        results_lock = threading.Lock()

        # Use a worker pool. Each worker assigned a slot (0..n_workers-1).
        # Items are pulled from a shared queue.
        import queue
        work_queue: queue.Queue = queue.Queue()
        for item in work:
            work_queue.put(item)

        def worker(slot: int):
            local: List[ScenarioResult] = []
            while True:
                try:
                    sc, sym = work_queue.get_nowait()
                except queue.Empty:
                    return local
                logger.info("[slot %d] %s × %s …", slot, sc.name, sym)
                t0 = time.time()
                res = self.run_one(sc, sym, slot=slot)
                local.append(res)
                logger.info(
                    "[slot %d]   finished in %.1fs  trades=%d  pnl=%+.2f%%",
                    slot, time.time() - t0, res.trades,
                    (res.final_equity - self.initial_equity)
                        / self.initial_equity * 100.0,
                )
                with results_lock:
                    results.append(res)
                    done = len(results)
                    for w in self.live_writers:
                        w.set_overall_progress(done, total)
                    self._persist_partial(results)
            return local

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(worker, i) for i in range(n_workers)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("worker crashed: %s", exc)
                    traceback.print_exc()

        for w in self.live_writers:
            w.shutdown()
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _persist_partial(self, results: List[ScenarioResult]) -> None:
        df = pd.DataFrame([r.to_dict() for r in results])
        df.to_csv(self.output_dir / "_results_so_far.csv", index=False)

    def write_report(self, results: List[ScenarioResult]) -> Path:
        df = pd.DataFrame([r.to_dict() for r in results])
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        csv_path = self.output_dir / f"scenario_report_{ts}.csv"
        md_path = self.output_dir / f"scenario_report_{ts}.md"
        df.to_csv(csv_path, index=False)

        lines = [
            f"# Scenario Battery Report — {ts}",
            "",
            f"Total runs:    **{len(results)}**",
            f"Failed runs:   **{int(df['failed'].sum())}**",
            f"Scenarios:     **{df['scenario'].nunique()}**",
            f"Symbols:       **{df['symbol'].nunique()}**",
            "",
        ]

        # Overall stats
        clean = df[~df["failed"]]
        if not clean.empty:
            strat_ret = (clean["final_equity"] - self.initial_equity) / self.initial_equity * 100.0
            lines += [
                "## Overall",
                "",
                f"- Mean strategy return:    **{strat_ret.mean():+.2f}%**",
                f"- Mean benchmark return:   **{clean['benchmark_return_pct'].mean():+.2f}%**",
                f"- Mean relative-to-bench:  **{clean['relative_vs_benchmark_pct'].mean():+.2f}%**",
                f"- Mean trades per run:     **{clean['trades'].mean():.1f}**",
                f"- Mean win rate:           **{clean['win_rate'].mean() * 100:.1f}%**",
                f"- Mean Sharpe:             **{clean['sharpe'].mean():.2f}**",
                f"- Mean max drawdown:       **{clean['max_drawdown_pct'].mean() * 100:.2f}%**",
                "",
            ]

            # By category
            lines += ["## By category", ""]
            for cat, sub in clean.groupby("category"):
                sret = (sub["final_equity"] - self.initial_equity) / self.initial_equity * 100.0
                lines += [
                    f"### {cat.upper()}  ({len(sub)} runs)",
                    "",
                    f"- Mean strategy return: **{sret.mean():+.2f}%**",
                    f"- Mean benchmark:       **{sub['benchmark_return_pct'].mean():+.2f}%**",
                    f"- Mean vs benchmark:    **{sub['relative_vs_benchmark_pct'].mean():+.2f}%**",
                    f"- Mean trades:          **{sub['trades'].mean():.1f}**",
                    f"- Mean Sharpe:          **{sub['sharpe'].mean():.2f}**",
                    "",
                ]

            # Per-scenario table
            lines += ["## Per-scenario detail", "", "| scenario | symbol | trades | win% | strat % | bench % | rel % | maxDD% | sharpe |",
                       "|---|---|---|---|---|---|---|---|---|"]
            for _, row in clean.iterrows():
                sret = (row["final_equity"] - self.initial_equity) / self.initial_equity * 100.0
                lines.append(
                    f"| {row['scenario']} | {row['symbol']} | "
                    f"{int(row['trades'])} | {row['win_rate'] * 100:.0f}% | "
                    f"{sret:+.2f}% | {row['benchmark_return_pct']:+.2f}% | "
                    f"{row['relative_vs_benchmark_pct']:+.2f}% | "
                    f"{row['max_drawdown_pct'] * 100:.1f}% | "
                    f"{row['sharpe']:.2f} |"
                )

        # Failures
        failures = df[df["failed"]]
        if not failures.empty:
            lines += ["", "## Failures", ""]
            for _, row in failures.iterrows():
                lines.append(f"- `{row['scenario']}` × `{row['symbol']}`: {row['error']}")

        md_path.write_text("\n".join(lines))
        logger.info("Wrote scenario report: %s", md_path)
        logger.info("Wrote CSV:             %s", csv_path)
        return md_path
