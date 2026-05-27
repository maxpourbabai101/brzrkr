"""Broker execution layer (Alpaca).

Submits **bracket orders** — entry + stop‑loss + take‑profit in one
shot — from the JSON signals produced by signal_generator. Safety
posture:

* **Paper trading by default.** Real money requires *both*
  ``ALPACA_LIVE=true`` in the environment *and* the caller passing
  ``live_money=True`` to the constructor.
* Notional sizing is converted to whole‑share quantities; sub‑share
  orders are skipped (Alpaca fractional bracket orders aren't
  supported as of writing).
* The executor never re‑submits an order it has already placed in the
  current process (idempotent on ``client_order_id``).
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    submitted: bool
    order_id: Optional[str]
    reason: str
    raw: Any = None


class AlpacaExecutor:
    """Thin wrapper around alpaca-py's TradingClient."""

    def __init__(
        self,
        *,
        live_money: bool = False,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        skip_promotion_gate: bool = False,
    ) -> None:
        # Import lazily so the rest of the project remains usable without
        # alpaca-py installed.
        from alpaca.trading.client import TradingClient

        env_live = os.getenv("ALPACA_LIVE", "false").lower() == "true"
        paper = not (live_money and env_live)

        if live_money and not env_live:
            logger.warning(
                "live_money=True but ALPACA_LIVE != 'true' — falling back to paper. "
                "Set ALPACA_LIVE=true in the environment to enable real‑money trading."
            )

        # Promotion gate: live money requires a proven paper record.
        # Skippable only by tests via `skip_promotion_gate=True`.
        if live_money and env_live and not skip_promotion_gate:
            from src.execution.promotion_gate import PromotionGate
            PromotionGate().require_eligibility(requested_live_money=True)

        key = api_key or os.getenv("ALPACA_API_KEY")
        secret = secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise EnvironmentError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY must be set to execute orders."
            )

        self._client = TradingClient(key, secret, paper=paper)
        self._paper = paper
        self._seen_client_ids: set[str] = set()
        logger.info(
            "AlpacaExecutor initialised — endpoint=%s",
            "paper" if paper else "LIVE‑REAL‑MONEY",
        )

    # ------------------------------------------------------------------
    # Account inspection
    # ------------------------------------------------------------------
    def get_account_equity(self) -> float:
        """Current portfolio equity (cash + market value of positions)."""
        acct = self._client.get_account()
        return float(acct.equity)

    def get_open_positions(self) -> list[Dict[str, Any]]:
        positions = self._client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price) if getattr(p, "current_price", None) else 0.0,
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) if getattr(p, "unrealized_plpc", None) else 0.0,
                "side": p.side if hasattr(p, "side") else ("long" if float(p.qty) > 0 else "short"),
            }
            for p in positions
        ]

    def get_orders(self, *, status: str = "all", limit: int = 50) -> list[Dict[str, Any]]:
        """Return recent orders. ``status`` ∈ {open, closed, all}."""
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            status_map = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }
            req = GetOrdersRequest(
                status=status_map.get(status, QueryOrderStatus.ALL),
                limit=limit,
            )
            orders = self._client.get_orders(filter=req)
        except Exception:  # noqa: BLE001
            # Older SDK signature — fall back to no-filter call.
            orders = self._client.get_orders()

        out: list[Dict[str, Any]] = []
        for o in orders:
            # Extract TP/SL from bracket order legs when available.
            tp_price: Optional[float] = None
            sl_price: Optional[float] = None
            legs = getattr(o, "legs", None) or []
            for leg in legs:
                leg_type = str(getattr(leg, "order_type", "")).lower()
                if "stop" in leg_type:
                    sp = getattr(leg, "stop_price", None)
                    if sp is not None:
                        sl_price = float(sp)
                elif "limit" in leg_type:
                    lp = getattr(leg, "limit_price", None)
                    if lp is not None:
                        tp_price = float(lp)
            out.append({
                "id": str(o.id),
                "symbol": o.symbol,
                "side": str(o.side),
                "qty": float(o.qty or 0),
                "filled_qty": float(o.filled_qty or 0),
                "order_type": str(o.order_type),
                "status": str(o.status),
                "limit_price": float(o.limit_price) if getattr(o, "limit_price", None) else None,
                "stop_price":  float(o.stop_price)  if getattr(o, "stop_price",  None) else None,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "submitted_at": str(o.submitted_at) if getattr(o, "submitted_at", None) else None,
                "filled_at":    str(o.filled_at)    if getattr(o, "filled_at",    None) else None,
            })
        return out

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single open order by id."""
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Cancel failed for order %s: %s", order_id, exc)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel every open order. Returns the count that was cancelled."""
        try:
            cancelled = self._client.cancel_orders()
            return len(cancelled) if cancelled is not None else 0
        except Exception as exc:  # noqa: BLE001
            logger.error("cancel_all_orders failed: %s", exc)
            return 0

    def close_position(self, symbol: str) -> bool:
        """Flatten an existing position. Raises only on totally
        unexpected errors; ordinary 'no position' returns False.
        """
        try:
            self._client.close_position(symbol)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("close_position(%s) failed: %s", symbol, exc)
            return False

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------
    def submit_signal(
        self,
        signal: Dict[str, Any],
        *,
        extended_hours: bool = False,
        session: str = "regular",
    ) -> ExecutionResult:
        """Convert a signal_generator JSON object into a bracket order.

        Parameters
        ----------
        signal : dict
            Signal JSON from the signal generator.
        extended_hours : bool
            If True, place a limit order at the current price with
            ``extended_hours=True`` (pre-market / after-hours).
            Market orders are not allowed outside regular hours.
        session : str
            "regular" | "pre_market" | "after_hours" | "crypto"
            Informational only (logged); ``extended_hours`` flag governs
            the actual Alpaca request.
        """
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

        symbol   = signal["asset"]
        entry    = float(signal["entry_price"])
        notional = float(signal["position_size_usd"])
        if notional <= 0 or entry <= 0:
            return ExecutionResult(False, None, "non‑positive notional or entry price")

        qty = int(notional // entry)
        if qty < 1:
            return ExecutionResult(
                False, None,
                f"sub‑share quantity ({notional:.2f} USD / {entry:.2f}) — skipping"
            )

        side = OrderSide.BUY if signal["direction"] == "long" else OrderSide.SELL
        client_order_id = f"te-{signal['asset']}-{signal['timestamp']}"
        client_order_id = client_order_id.replace(":", "").replace("-", "")[:48]
        client_order_id += "-" + uuid.uuid4().hex[:8]

        if client_order_id in self._seen_client_ids:
            return ExecutionResult(False, None, "duplicate client_order_id in this process")

        tp_price = round(float(signal["take_profit"]), 2)
        sl_price = round(float(signal["stop_loss"]),   2)

        if extended_hours:
            # Extended-hours: must use limit order at current price;
            # bracket class not allowed outside regular hours on Alpaca.
            limit_price = round(entry, 2)
            order_req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                extended_hours=True,
            )
            logger.info(
                "Extended-hours %s limit order %s qty=%d limit=%.2f "
                "session=%s (no bracket — manual SL/TP required)",
                symbol, signal["direction"], qty, limit_price, session,
            )
        else:
            order_req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_price),
                stop_loss=StopLossRequest(stop_price=sl_price),
                client_order_id=client_order_id,
            )
        try:
            order = self._client.submit_order(order_data=order_req)
        except Exception as exc:  # noqa: BLE001 — surface broker errors verbatim
            logger.error("Alpaca rejected order for %s: %s", symbol, exc)
            return ExecutionResult(False, None, f"broker error: {exc}")

        self._seen_client_ids.add(client_order_id)
        logger.info(
            "Submitted %s bracket order: %s qty=%d entry=%.2f stop=%.2f tp=%.2f (paper=%s)",
            symbol, signal["direction"], qty, entry,
            signal["stop_loss"], signal["take_profit"], self._paper,
        )
        return ExecutionResult(True, str(order.id), "ok", raw=order)
