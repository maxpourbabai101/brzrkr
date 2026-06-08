"""
TrailingStopManager — ATR-based trailing stops for open long positions.

Every tick:
  1. Get current price for each open position
  2. Update the high-water mark (highest price seen since entry)
  3. Compute new trailing stop = HWM - atr_mult × ATR(14)
  4. If new stop > current stop + min_move: cancel old stop, place new one

This lets winners run while protecting against reversals.
Minimum update step of $0.75 prevents API rate-limit abuse.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MIN_STOP_MOVE  = 0.75   # only update Alpaca stop if it moves by $0.75+
ATR_MULTIPLIER = 2.0    # stop = HWM - 2 × ATR


class TrailingStopManager:
    """Manages trailing stops for open positions."""

    def __init__(self, atr_mult: float = ATR_MULTIPLIER,
                 min_move: float = MIN_STOP_MOVE) -> None:
        self.atr_mult = atr_mult
        self.min_move = min_move
        self._hwm: Dict[str, float] = {}   # symbol → highest price seen

    def update_all(
        self,
        executor,
        positions: list[Dict[str, Any]],
        prices_map: Dict[str, pd.DataFrame],
        journal_entries: list[Dict[str, Any]],
    ) -> None:
        """Call every tick to update trailing stops for all open long positions."""
        journal_by_sym = {e["symbol"]: e for e in journal_entries
                          if e.get("status") == "open"}

        for pos in positions:
            symbol    = pos.get("symbol", "")
            side      = str(pos.get("side", "long")).lower()
            curr_price = float(pos.get("current_price") or pos.get("avg_entry_price") or 0)
            qty       = abs(int(float(pos.get("qty", 0))))

            if "long" not in side or curr_price <= 0 or qty < 1:
                continue   # only trail longs for now

            # Update high-water mark
            prev_hwm  = self._hwm.get(symbol, curr_price)
            new_hwm   = max(prev_hwm, curr_price)
            self._hwm[symbol] = new_hwm

            # Get ATR from price data
            prices = prices_map.get(symbol)
            if prices is None or prices.empty or len(prices) < 14:
                continue

            close = prices["close"].astype(float)
            high  = prices["high"].astype(float)
            low   = prices["low"].astype(float)

            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1] or 0)

            if atr <= 0:
                continue

            new_stop = round(new_hwm - self.atr_mult * atr, 2)

            # Get current stop from journal
            journal = journal_by_sym.get(symbol)
            if journal is None:
                continue
            current_stop = float(journal.get("stop_loss") or 0)

            if current_stop <= 0:
                continue

            # Only update if stop moved up meaningfully
            if new_stop <= current_stop + self.min_move:
                logger.debug(
                    "TrailingStop %s: new_stop=%.2f not enough above current=%.2f — skip",
                    symbol, new_stop, current_stop,
                )
                continue

            logger.info(
                "TrailingStop %s: HWM=%.2f ATR=%.2f → stop moved %.2f→%.2f",
                symbol, new_hwm, atr, current_stop, new_stop,
            )

            # Cancel existing stop order and place new one
            try:
                open_orders = executor.get_orders(status="open", limit=100)
                for order in open_orders:
                    if order.get("symbol") == symbol:
                        otype = str(order.get("order_type", "")).lower()
                        if "stop" in otype:
                            executor.cancel_order(order["id"])
                            logger.debug("Cancelled old stop order %s for %s",
                                         order["id"], symbol)
                            break

                # Place new trailing stop
                executor.protect_position(
                    symbol=symbol,
                    qty=qty,
                    stop_price=new_stop,
                    take_profit_price=float(journal.get("take_profit") or new_hwm * 1.05),
                    direction="long",
                )

                # Update journal stop price
                try:
                    from src.learning.trade_journal import TradeJournal
                    j = TradeJournal()
                    all_records = j._load_all()
                    for rec in all_records:
                        if rec.get("symbol") == symbol and rec.get("status") == "open":
                            rec["stop_loss"] = new_stop
                    j._save_all(all_records)
                except Exception as _je:
                    logger.debug("TrailingStop journal update failed: %s", _je)

            except Exception as exc:
                logger.warning("TrailingStop update FAILED for %s: %s", symbol, exc)
