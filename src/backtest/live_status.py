"""LiveStatusWriter — atomic JSON status file for in-progress simulations.

A single ``data/scenario_runs/_live.json`` is updated as the
``ScenarioRunner`` works through scenarios. The dashboard polls it
every few seconds to show live progress + equity curve.

Atomic write pattern (tmp + rename) so the dashboard never reads a
half-written file even though writes happen from a worker thread.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/scenario_runs/_live.json")


@dataclass
class LiveStatusWriter:
    """Thread-safe writer for live simulation status."""

    path: Path = field(default_factory=lambda: DEFAULT_PATH)
    history_max: int = 360         # equity points to keep in memory/file
    write_throttle_s: float = 0.15 # minimum seconds between disk writes

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._equity_history: Deque[float] = deque(maxlen=self.history_max)
        self._last_write_ts: float = 0.0
        self._state: Dict[str, Any] = {
            "active": False,
            "scenario": None,
            "symbol": None,
            "category": None,
            "bars_total": 0,
            "bars_processed": 0,
            "current_equity": 0.0,
            "initial_equity": 0.0,
            "trades_so_far": 0,
            "open_trade": None,
            "equity_history": [],
            "started_at": None,
            "scenarios_done": 0,
            "scenarios_total": 0,
            "updated_at": None,
        }

    # ------------------------------------------------------------------
    # Scenario lifecycle
    # ------------------------------------------------------------------
    def set_overall_progress(self, done: int, total: int) -> None:
        with self._lock:
            self._state["scenarios_done"] = int(done)
            self._state["scenarios_total"] = int(total)
            self._flush(force=True)

    def start(
        self,
        *,
        scenario: str,
        symbol: str,
        category: str,
        bars_total: int,
        initial_equity: float,
    ) -> None:
        with self._lock:
            self._equity_history.clear()
            self._equity_history.append(initial_equity)
            self._state.update({
                "active": True,
                "scenario": scenario,
                "symbol": symbol,
                "category": category,
                "bars_total": int(bars_total),
                "bars_processed": 0,
                "current_equity": float(initial_equity),
                "initial_equity": float(initial_equity),
                "trades_so_far": 0,
                "open_trade": None,
                "equity_history": list(self._equity_history),
                "started_at": datetime.now(timezone.utc).isoformat(),
            })
            self._flush(force=True)

    def update(
        self,
        *,
        bars_processed: int,
        current_equity: float,
        trades_so_far: Optional[int] = None,
        open_trade: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._equity_history.append(float(current_equity))
            self._state["bars_processed"] = int(bars_processed)
            self._state["current_equity"] = float(current_equity)
            self._state["equity_history"] = list(self._equity_history)
            if trades_so_far is not None:
                self._state["trades_so_far"] = int(trades_so_far)
            if open_trade is not None:
                self._state["open_trade"] = open_trade
            else:
                self._state["open_trade"] = None
            self._flush()

    def finish(self) -> None:
        with self._lock:
            self._state["active"] = False
            self._state["bars_processed"] = self._state["bars_total"]
            self._flush(force=True)

    def shutdown(self) -> None:
        """Mark the entire battery as done."""
        with self._lock:
            self._state["active"] = False
            self._state["scenarios_done"] = self._state.get(
                "scenarios_total", 0)
            self._flush(force=True)

    # ------------------------------------------------------------------
    # Reading (for the dashboard)
    # ------------------------------------------------------------------
    @classmethod
    def read(cls, path: Path = DEFAULT_PATH) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _flush(self, *, force: bool = False) -> None:
        """Write to disk via tmp + rename. Throttled unless force=True."""
        now = time.monotonic()
        if not force and (now - self._last_write_ts) < self.write_throttle_s:
            return
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(self._state, default=str))
            os.replace(tmp, self.path)
            self._last_write_ts = now
        except Exception as exc:  # noqa: BLE001
            logger.debug("live status write failed: %s", exc)
