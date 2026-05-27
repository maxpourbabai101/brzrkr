"""Broker registry — switchable execution backends.

The agent / dashboard / trade.py only need an object with this
interface::

    .get_account_equity() -> float
    .get_open_positions() -> list[dict]
    .get_orders(status='all', limit=50) -> list[dict]
    .submit_signal(signal_dict) -> ExecutionResult
    .cancel_order(order_id) -> bool
    .cancel_all_orders() -> int
    .close_position(symbol) -> bool

Use :func:`get_broker` to instantiate the right backend by name. The
promotion gate is still enforced for any live-money request, regardless
of which broker you pick.

Currently shipped:

* ``alpaca``  — fully implemented (paper + live), thin wrapper around
  ``AlpacaExecutor``.
* ``ibkr``    — placeholder / stub. Wires up ``ib_insync`` if installed
  but most methods raise ``NotImplementedError`` until you implement
  them. Provided so you can switch brokers later by implementing one
  class, not touching the agent / dashboard.
* ``paper_only`` — pure in-memory simulator. Useful for tests / demos
  where you want the full pipeline to "trade" without an exchange.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type (mirrors the one in broker.py)
# ---------------------------------------------------------------------------
@dataclass
class ExecutionResult:
    submitted: bool
    order_id: Optional[str]
    reason: str
    raw: Any = None


# ---------------------------------------------------------------------------
# In-memory simulator — useful for tests and offline demos
# ---------------------------------------------------------------------------
@dataclass
class InMemoryBroker:
    """Self-contained simulator. No network, no real money."""

    starting_equity: float = 100_000.0
    name: str = "paper_only"

    _equity: float = field(init=False)
    _positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _orders: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._equity = float(self.starting_equity)

    # --- read methods -------------------------------------------------
    def get_account_equity(self) -> float:
        # Mark-to-market = cash + sum of position market values.
        mv = sum(p["market_value"] for p in self._positions.values())
        return self._equity + mv

    def get_open_positions(self) -> List[Dict[str, Any]]:
        return list(self._positions.values())

    def get_orders(self, *, status: str = "all", limit: int = 50
                   ) -> List[Dict[str, Any]]:
        out = self._orders[-limit:]
        if status == "open":
            out = [o for o in out
                   if o["status"] in ("accepted", "pending_new", "new",
                                      "partially_filled")]
        elif status == "closed":
            out = [o for o in out
                   if o["status"] in ("filled", "canceled", "rejected")]
        return list(out)

    # --- write methods ------------------------------------------------
    def submit_signal(self, signal: Dict[str, Any]) -> ExecutionResult:
        symbol = signal["asset"]
        entry = float(signal["entry_price"])
        notional = float(signal["position_size_usd"])
        if notional <= 0 or entry <= 0:
            return ExecutionResult(False, None, "non-positive notional/entry")
        qty = int(notional // entry)
        if qty < 1:
            return ExecutionResult(False, None, "sub-share quantity")

        order_id = f"sim-{uuid.uuid4().hex[:10]}"
        side = "buy" if signal["direction"] == "long" else "sell"
        # Immediate fill at entry price.
        self._orders.append({
            "id": order_id, "symbol": symbol, "side": side,
            "qty": qty, "filled_qty": qty,
            "order_type": "market", "status": "filled",
            "limit_price": None, "stop_price": float(signal["stop_loss"]),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "filled_at": datetime.now(timezone.utc).isoformat(),
        })
        sign = 1 if side == "buy" else -1
        self._equity -= sign * qty * entry  # cash spent / collected
        existing = self._positions.get(symbol)
        if existing:
            existing["qty"] += sign * qty
            existing["market_value"] = existing["qty"] * entry
        else:
            self._positions[symbol] = {
                "symbol": symbol, "qty": sign * qty,
                "avg_entry_price": entry, "current_price": entry,
                "market_value": sign * qty * entry,
                "unrealized_pl": 0.0, "unrealized_plpc": 0.0,
                "side": "long" if sign > 0 else "short",
            }
        return ExecutionResult(True, order_id, "ok")

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o["id"] == order_id and o["status"] in ("accepted", "new"):
                o["status"] = "canceled"
                return True
        return False

    def cancel_all_orders(self) -> int:
        n = 0
        for o in self._orders:
            if o["status"] in ("accepted", "new", "pending_new",
                                "partially_filled"):
                o["status"] = "canceled"
                n += 1
        return n

    def close_position(self, symbol: str) -> bool:
        p = self._positions.pop(symbol, None)
        if p is None:
            return False
        self._equity += p["market_value"]
        return True


# ---------------------------------------------------------------------------
# IBKR stub — wires the registry but defers real work to a future PR
# ---------------------------------------------------------------------------
@dataclass
class IBKRBroker:
    """Placeholder. Implement against `ib_insync`'s IB() and IB.placeOrder()
    when you graduate from Alpaca.
    """
    host: str = "127.0.0.1"
    port: int = 7497            # 7497 = paper TWS, 7496 = live TWS
    client_id: int = 7
    name: str = "ibkr"

    def __post_init__(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise ImportError(
                "ib_insync not installed. `pip install ib_insync` "
                "and run Trader Workstation / IB Gateway before using IBKR."
            ) from exc
        # Intentionally NOT connecting here — implement when you're ready.
        logger.warning(
            "IBKRBroker instantiated but not implemented. "
            "Methods will raise NotImplementedError until you wire them up."
        )
        self._ib = None

    def get_account_equity(self) -> float:
        raise NotImplementedError("IBKR equity not implemented. See ib_insync docs.")

    def get_open_positions(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("IBKR positions not implemented.")

    def get_orders(self, *, status: str = "all", limit: int = 50
                   ) -> List[Dict[str, Any]]:
        raise NotImplementedError("IBKR orders not implemented.")

    def submit_signal(self, signal: Dict[str, Any]) -> ExecutionResult:
        raise NotImplementedError("IBKR submit_signal not implemented.")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("IBKR cancel_order not implemented.")

    def cancel_all_orders(self) -> int:
        raise NotImplementedError("IBKR cancel_all_orders not implemented.")

    def close_position(self, symbol: str) -> bool:
        raise NotImplementedError("IBKR close_position not implemented.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def get_broker(name: str, *, live_money: bool = False, **kwargs):
    """Return a broker instance by name.

    The promotion gate is enforced inside each constructor that
    supports live money — switching brokers does NOT bypass it.
    """
    name = name.lower()
    if name == "alpaca":
        from src.execution.broker import AlpacaExecutor
        return AlpacaExecutor(live_money=live_money, **kwargs)
    if name == "ibkr":
        # Live-money guard for IBKR will live here once the broker is real.
        if live_money:
            from src.execution.promotion_gate import PromotionGate
            PromotionGate().require_eligibility(requested_live_money=True)
        return IBKRBroker(**kwargs)
    if name in ("paper_only", "simulator", "sim"):
        return InMemoryBroker(**kwargs)
    raise ValueError(
        f"Unknown broker {name!r}. Available: alpaca, ibkr, paper_only."
    )
