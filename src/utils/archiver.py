"""BatchArchiver — compact long-running backtest output.

After every scenario battery finishes, the runner leaves per-scenario
trade ledgers and equity curves in
``data/scenario_runs/<scenario>/<symbol>/``. Over many batteries these
files pile up: ~100 KB per battery × N batteries / day = a few MB/day.

The archiver consolidates each batch:

1. **Append every batch's trades** to ``data/all_trades.csv`` (master
   append-only ledger the profit optimiser reads).
2. **Append the batch summary** to ``data/scenario_summary_log.csv``
   (one row per scenario × symbol per batch).
3. **Tar+gz the per-scenario detail directories** into
   ``data/archive/batch_<timestamp>.tar.gz``.
4. **Delete the original per-scenario directories** once archived.
5. **Trim old archives** beyond ``keep_archives``.

Result: the dashboard's live data is always a single small master file,
historical detail is preserved as tarballs, and disk usage stays
bounded.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ArchiveStats:
    batch_id: str
    scenarios_archived: int
    trades_appended: int
    archive_path: Optional[Path] = None
    archive_size_kb: float = 0.0
    deleted_directories: int = 0


@dataclass
class BatchArchiver:
    scenario_dir: Path = field(default_factory=lambda: Path("data/scenario_runs"))
    archive_dir: Path = field(default_factory=lambda: Path("data/archive"))
    master_trades: Path = field(default_factory=lambda: Path("data/all_trades.csv"))
    summary_log: Path = field(default_factory=lambda: Path("data/scenario_summary_log.csv"))
    keep_archives: int = 30                # trim beyond this many tarballs
    keep_recent_uncompressed: int = 0      # set >0 to NOT archive the last N batches

    def __post_init__(self) -> None:
        self.scenario_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.master_trades.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def archive_current_batch(self, *, batch_id: Optional[str] = None
                               ) -> ArchiveStats:
        """Aggregate + compress every scenario directory currently under
        ``scenario_dir``. Returns stats about what was processed."""
        batch_id = batch_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ")
        stats = ArchiveStats(batch_id=batch_id, scenarios_archived=0,
                              trades_appended=0)

        # Find all per-scenario directories (children of scenario_dir
        # that are themselves dirs).
        scenario_subdirs = [
            p for p in sorted(self.scenario_dir.iterdir())
            if p.is_dir() and not p.name.startswith("_")
               and not p.name.startswith("archive")
        ]
        if not scenario_subdirs:
            logger.info("No scenario directories to archive.")
            return stats

        # 1. Aggregate trades into the master file.
        trades_frames: List[pd.DataFrame] = []
        summary_rows: List[Dict] = []
        for scen_dir in scenario_subdirs:
            for sym_dir in scen_dir.iterdir():
                if not sym_dir.is_dir():
                    continue
                trades_csv = sym_dir / "trades.csv"
                equity_csv = sym_dir / "equity_curve.csv"
                if trades_csv.exists():
                    try:
                        df = pd.read_csv(trades_csv)
                        if not df.empty:
                            df["scenario"] = scen_dir.name
                            df["symbol"] = sym_dir.name
                            df["batch_id"] = batch_id
                            trades_frames.append(df)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("skip trades %s: %s", trades_csv, exc)
                # Summary: final equity + bar count
                if equity_csv.exists():
                    try:
                        eq = pd.read_csv(equity_csv)
                        if not eq.empty:
                            final_eq = float(eq.iloc[-1, 1])
                            bars = int(len(eq))
                            summary_rows.append({
                                "batch_id": batch_id,
                                "scenario": scen_dir.name,
                                "symbol": sym_dir.name,
                                "bars": bars,
                                "final_equity": final_eq,
                            })
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("skip equity %s: %s", equity_csv, exc)

        if trades_frames:
            master = pd.concat(trades_frames, ignore_index=True)
            stats.trades_appended = len(master)
            # Append (write header only if the file doesn't exist yet)
            write_header = not self.master_trades.exists()
            master.to_csv(self.master_trades, mode="a",
                           header=write_header, index=False)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            write_header = not self.summary_log.exists()
            summary_df.to_csv(self.summary_log, mode="a",
                               header=write_header, index=False)

        # 2. Tar + gzip the per-scenario detail.
        archive_path = self.archive_dir / f"batch_{batch_id}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for scen_dir in scenario_subdirs:
                tar.add(scen_dir, arcname=scen_dir.name)
                stats.scenarios_archived += 1
        stats.archive_path = archive_path
        stats.archive_size_kb = archive_path.stat().st_size / 1024.0

        # 3. Delete original per-scenario directories.
        for scen_dir in scenario_subdirs:
            try:
                shutil.rmtree(scen_dir)
                stats.deleted_directories += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("delete %s failed: %s", scen_dir, exc)

        # 4. Trim old archives.
        self.trim_old_archives()

        logger.info(
            "Archived batch %s: %d scenarios, %d trades, %.1f KB → %s",
            batch_id, stats.scenarios_archived, stats.trades_appended,
            stats.archive_size_kb, archive_path,
        )
        return stats

    # ------------------------------------------------------------------
    def trim_old_archives(self) -> int:
        """Delete tarballs beyond ``keep_archives`` (oldest first).
        Returns count deleted."""
        archives = sorted(self.archive_dir.glob("batch_*.tar.gz"))
        if len(archives) <= self.keep_archives:
            return 0
        to_delete = archives[: len(archives) - self.keep_archives]
        n = 0
        for p in to_delete:
            try:
                p.unlink()
                n += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("trim %s failed: %s", p, exc)
        if n:
            logger.info("Trimmed %d old archive(s).", n)
        return n

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        """Disk usage report."""
        master_kb = (self.master_trades.stat().st_size / 1024.0
                     if self.master_trades.exists() else 0)
        summary_kb = (self.summary_log.stat().st_size / 1024.0
                      if self.summary_log.exists() else 0)
        archives = list(self.archive_dir.glob("batch_*.tar.gz"))
        archives_total_kb = sum(p.stat().st_size for p in archives) / 1024.0
        # Live scenario dir (not yet archived)
        live_total = 0
        if self.scenario_dir.exists():
            for p in self.scenario_dir.rglob("*"):
                if p.is_file():
                    try:
                        live_total += p.stat().st_size
                    except Exception:
                        pass
        return {
            "master_trades_kb": round(master_kb, 1),
            "summary_log_kb": round(summary_kb, 1),
            "archives_count": len(archives),
            "archives_total_kb": round(archives_total_kb, 1),
            "live_scenario_dir_kb": round(live_total / 1024.0, 1),
        }
