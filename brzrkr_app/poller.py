"""BrokerPoller — background thread that keeps broker state fresh.

Pushes snapshots into a queue every N seconds; the main window reads
the queue from the Tk event loop without blocking.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from queue import Queue
from typing import Optional

logger = logging.getLogger(__name__)


class BrokerPoller(threading.Thread):
    def __init__(self, queue: Queue, interval: float = 8.0) -> None:
        super().__init__(daemon=True)
        self.queue = queue
        self.interval = interval
        self._stop = threading.Event()
        self._executor = None
        self._fast_next = False

    def stop(self) -> None:
        self._stop.set()

    def trigger(self) -> None:
        self._fast_next = True

    def run(self) -> None:
        while not self._stop.is_set():
            self.queue.put(self._snapshot())
            for _ in range(int(self.interval * 10)):
                if self._stop.is_set() or self._fast_next:
                    self._fast_next = False
                    break
                time.sleep(0.1)

    def _snapshot(self) -> dict:
        try:
            from src.execution.broker import AlpacaExecutor
            if self._executor is None:
                self._executor = AlpacaExecutor(live_money=False)
            return {
                "ok": True,
                "equity": self._executor.get_account_equity(),
                "positions": self._executor.get_open_positions(),
                "orders": self._executor.get_orders(status="all", limit=50),
                "paper": self._executor._paper,
                "ts": datetime.now(timezone.utc),
                "executor": self._executor,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": str(exc),
                "ts": datetime.now(timezone.utc),
                "equity": 0.0,
                "positions": [],
                "orders": [],
                "paper": True,
                "executor": None,
            }
