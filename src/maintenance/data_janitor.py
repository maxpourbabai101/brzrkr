"""DataJanitor — self-cleaning agent for the trading system.

Runs once per day (called by TradingAgent on day rollover, or standalone
via ``python -m src.maintenance.data_janitor``).

What it cleans                  Retention policy
───────────────────────────────────────────────────────────────────────
data/signals/*.json             Keep last 90 days; never delete today's
data/market_vectors/*.npz       Keep newest 14; delete older snapshots
data/archive/*.tar.gz           Keep newest 30 batch archives
data/scenario_runs/**/*.json    Keep newest 3 per scenario×symbol dir
data/scenario_runs/**/*.csv     Keep newest 3 per scenario×symbol dir
data/scenario_runs/*_report_*   Keep newest 5 top-level reports
data/scenario_runs/_results_*   Always delete — replaced each run
data/_live*.json                Keep only (re-created every run)

NEVER touches:
  data/trade_journal.jsonl      Live trade log — sacred
  data/model_accuracy.json      Online learning — sacred
  data/learned_params.json      Strategy parameters — sacred
  data/strategy_state.json      Agent state — sacred
  data/postmortems.jsonl        Post-mortems — sacred
  data/track_record.jsonl       Track record — sacred
  data/regime_cache.json        Regime state — sacred
  data/all_trades.csv           Full trade export — sacred
  data/reports/*.md             System reports — sacred
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Files that must never be deleted regardless of age.
# ---------------------------------------------------------------------------
_SACRED: frozenset = frozenset([
    "trade_journal.jsonl",
    "model_accuracy.json",
    "learned_params.json",
    "strategy_state.json",
    "postmortems.jsonl",
    "track_record.jsonl",
    "regime_cache.json",
    "all_trades.csv",
    "scanner_attribution.json",
    "doctor_report.md",
])


@dataclass
class CleanupReport:
    """Summary returned by DataJanitor.run()."""
    files_deleted: int = 0
    bytes_freed: int = 0
    errors: List[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def __str__(self) -> str:
        mb = self.bytes_freed / (1024 * 1024)
        return (
            f"DataJanitor: deleted {self.files_deleted} files, "
            f"freed {mb:.1f} MB "
            f"({len(self.errors)} errors)"
        )


class DataJanitor:
    """Autonomous cleanup agent. Call .run() once per day.

    Parameters
    ----------
    data_dir:
        Root of the data directory (default: ``data/`` relative to CWD).
    signal_retention_days:
        How many calendar days of signal JSON files to keep (default 90).
    market_vector_keep:
        How many .npz snapshot files to keep (default 14, ~2 weeks).
    archive_keep:
        How many .tar.gz batch archives to keep (default 30).
    scenario_artifact_keep:
        How many JSON/CSV files to keep per scenario×symbol subdirectory
        (default 3 — keeps the last three runs for comparison).
    report_keep:
        How many top-level scenario report files to keep (default 5).
    dry_run:
        If True, log what *would* be deleted but don't actually delete.
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        *,
        signal_retention_days: int = 90,
        market_vector_keep: int = 14,
        archive_keep: int = 30,
        scenario_artifact_keep: int = 3,
        report_keep: int = 5,
        dry_run: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.signal_retention_days = signal_retention_days
        self.market_vector_keep = market_vector_keep
        self.archive_keep = archive_keep
        self.scenario_artifact_keep = scenario_artifact_keep
        self.report_keep = report_keep
        self.dry_run = dry_run
        self._report = CleanupReport()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> CleanupReport:
        """Execute all cleanup tasks and return a summary."""
        self._report = CleanupReport(
            started_at=datetime.now(timezone.utc).isoformat()
        )
        t0 = time.monotonic()

        tasks = [
            ("signals",          self._clean_signals),
            ("market_vectors",   self._clean_market_vectors),
            ("archives",         self._clean_archives),
            ("scenario_runs",    self._clean_scenario_runs),
        ]
        for name, fn in tasks:
            try:
                fn()
            except Exception as exc:
                msg = f"{name}: {exc}"
                logger.error("DataJanitor task failed — %s", msg)
                self._report.errors.append(msg)

        self._report.finished_at = datetime.now(timezone.utc).isoformat()
        elapsed = time.monotonic() - t0
        logger.info(
            "%s  (took %.1fs)%s",
            self._report,
            elapsed,
            "  [DRY RUN]" if self.dry_run else "",
        )
        return self._report

    # ------------------------------------------------------------------
    # Individual cleanup tasks
    # ------------------------------------------------------------------
    def _clean_signals(self) -> None:
        """Delete signal JSON files older than signal_retention_days."""
        sig_dir = self.data_dir / "signals"
        if not sig_dir.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.signal_retention_days)
        for f in sig_dir.glob("*.json"):
            if self._is_sacred(f):
                continue
            # Filename pattern: SYMBOL_YYYYMMDDTHHMMSSz.json
            try:
                ts_part = f.stem.rsplit("_", 1)[-1]          # e.g. 20260527T120000Z
                ts = datetime.strptime(ts_part, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    self._delete(f)
            except (ValueError, IndexError):
                # Can't parse timestamp — fall back to file mtime
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    self._delete(f)

    def _clean_market_vectors(self) -> None:
        """Keep the newest market_vector_keep .npz files; delete the rest."""
        mv_dir = self.data_dir / "market_vectors"
        if not mv_dir.exists():
            return
        # latest.json is a metadata pointer — never touch it
        npz_files = sorted(
            mv_dir.glob("*.npz"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,               # newest first
        )
        for f in npz_files[self.market_vector_keep:]:
            if self._is_sacred(f):
                continue
            self._delete(f)

    def _clean_archives(self) -> None:
        """Keep the newest archive_keep .tar.gz files; delete the rest."""
        archive_dir = self.data_dir / "archive"
        if not archive_dir.exists():
            return
        gz_files = sorted(
            archive_dir.glob("*.tar.gz"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for f in gz_files[self.archive_keep:]:
            self._delete(f)

    def _clean_scenario_runs(self) -> None:
        """Clean scenario_runs directory:

        1. For each scenario/symbol/ leaf directory, keep only the newest
           scenario_artifact_keep JSON/CSV files.
        2. In the scenario_runs root, keep only the newest report_keep
           *_report_* files (md + csv pairs counted together by timestamp).
        3. Always delete _results_so_far.csv (it's a temp file).
        4. Delete _live*.json (recreated on each run start).
        """
        sr = self.data_dir / "scenario_runs"
        if not sr.exists():
            return

        # 3+4: temp files in the root
        for pattern in ("_results_so_far.csv", "_live*.json"):
            for f in sr.glob(pattern):
                self._delete(f)

        # 2: top-level report files — keep newest report_keep pairs
        report_files = sorted(
            list(sr.glob("*_report_*.md")) + list(sr.glob("*_report_*.csv")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for f in report_files[self.report_keep * 2:]:  # × 2 because md + csv pairs
            self._delete(f)

        # 1: per-scenario×symbol leaf dirs
        for leaf in sr.rglob("*/"):
            # Only act on directories that are at depth 2 (scenario/symbol/)
            try:
                rel = leaf.relative_to(sr)
                depth = len(rel.parts)
            except ValueError:
                continue
            if depth != 2:
                continue
            self._prune_leaf(leaf)

    def _prune_leaf(self, directory: Path) -> None:
        """Keep newest scenario_artifact_keep JSON/CSV files; delete older ones."""
        candidates = sorted(
            list(directory.glob("*.json")) + list(directory.glob("*.csv")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for f in candidates[self.scenario_artifact_keep:]:
            if self._is_sacred(f):
                continue
            self._delete(f)
        # If the directory is now empty (no sacred files left), remove it.
        remaining = list(directory.iterdir())
        if not remaining and not self.dry_run:
            try:
                directory.rmdir()
                logger.debug("Removed empty dir: %s", directory)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_sacred(self, path: Path) -> bool:
        return path.name in _SACRED

    def _delete(self, path: Path) -> None:
        size = 0
        try:
            size = path.stat().st_size
        except OSError:
            pass
        if self.dry_run:
            logger.info("[DRY RUN] would delete %s  (%d bytes)", path, size)
        else:
            try:
                path.unlink()
                self._report.files_deleted += 1
                self._report.bytes_freed += size
                logger.debug("Deleted %s  (%d bytes)", path, size)
            except OSError as exc:
                msg = f"Could not delete {path}: {exc}"
                logger.warning(msg)
                self._report.errors.append(msg)


# ---------------------------------------------------------------------------
# Singleton + convenience
# ---------------------------------------------------------------------------
_janitor: Optional[DataJanitor] = None


def get_janitor(data_dir: Path = Path("data"), **kwargs) -> DataJanitor:
    """Return a module-level singleton (safe to call from any thread)."""
    global _janitor
    if _janitor is None:
        _janitor = DataJanitor(data_dir=data_dir, **kwargs)
    return _janitor


# ---------------------------------------------------------------------------
# CLI entry point: python -m src.maintenance.data_janitor [--dry-run]
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="DataJanitor — clean up stale trading system files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting anything.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the data directory (default: data/)",
    )
    parser.add_argument(
        "--signal-days",
        type=int,
        default=90,
        help="Retain signal JSON files for this many days (default: 90)",
    )
    parser.add_argument(
        "--archive-keep",
        type=int,
        default=30,
        help="Number of archive batches to keep (default: 30)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    janitor = DataJanitor(
        data_dir=Path(args.data_dir),
        signal_retention_days=args.signal_days,
        archive_keep=args.archive_keep,
        dry_run=args.dry_run,
    )
    report = janitor.run()
    print(report)
    if report.errors:
        print(f"  Errors ({len(report.errors)}):")
        for e in report.errors:
            print(f"    • {e}")
    sys.exit(0 if not report.errors else 1)


if __name__ == "__main__":
    _cli()
