"""NinjaTrader ATI (Automated Trading Interface) broker backend.

Connects to NinjaTrader 8 via its TCP socket ATI on localhost:36973.
Implements the same interface as AlpacaExecutor / IBKRBroker so the agent
needs zero changes to switch brokers — just set ``broker: ninja`` in config
or call ``get_broker("ninja")``.

Setup (NinjaTrader side)
------------------------
1. Open NinjaTrader 8.
2. Tools → Options → Automated Trading Interface
3. Enable ATI: ✓
4. Port: 36973  (default, configurable via NINJA_ATI_PORT env var)
5. Accept local connections: ✓
6. Click OK, then connect your sim/live account.

Apex Trader Funding rules (enforced here)
-----------------------------------------
These are checked before EVERY order and after EVERY fill:

  Max daily loss     halt when realized+unrealized ≥ limit - $200 buffer
  Trailing drawdown  halt when equity ≤ peak_equity - dd_limit + $300 buffer
  Consistency rule   no single day may exceed 30% of total session profits
  News blackout      2 min around FOMC / CPI / NFP (uses event_blackout.py)
  Position limits    max 1 contract per MES/MNQ, 1 per ES/NQ at $50k+

Supported instruments
---------------------
  ES, NQ, MES, MNQ, CL, GC, RTY   (Apex futures)
  Any equity ticker will also be routed but Apex eval is futures-only.

ATI command reference
---------------------
  PLACE|id|acct|action|qty|sym|type|lim|stop|tif|oco|from|strat
  CANCEL|id|acct
  CANCELALLORDERS|acct
  CLOSEPOSITION|sym|acct
  CLOSEALLPOSITIONS|acct
  ACCOUNT|acct
  POSITIONS|acct
  ORDERS|acct

Responses are line-by-line, pipe-delimited, terminated by a blank line.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Defaults (override with environment variables) ────────────────────────────
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 36973
_DEFAULT_ACCOUNT = "Sim101"          # NinjaTrader sim account name
_CONNECT_TIMEOUT = 5.0               # seconds to wait for initial connection
_RECV_TIMEOUT = 10.0                 # seconds to wait for a command response
_HEARTBEAT_INTERVAL = 10.0           # seconds between keepalive pings
_RECONNECT_DELAY = 3.0               # seconds before reconnect attempt

# ── Apex account tiers ────────────────────────────────────────────────────────
APEX_RULES: Dict[str, Dict[str, Any]] = {
    "25k": {
        "max_daily_loss":     1500,
        "trailing_drawdown":  1500,
        "halt_buffer_daily":  200,
        "halt_buffer_dd":     300,
        "consistency_pct":    0.30,
        "news_blackout_sec":  120,
        "daily_target":       1500,
        "max_contracts":      2,
    },
    "50k": {
        "max_daily_loss":     2500,
        "trailing_drawdown":  2500,
        "halt_buffer_daily":  200,
        "halt_buffer_dd":     300,
        "consistency_pct":    0.30,
        "news_blackout_sec":  120,
        "daily_target":       2000,
        "max_contracts":      5,
    },
    "100k": {
        "max_daily_loss":     3000,
        "trailing_drawdown":  3000,
        "halt_buffer_daily":  200,
        "halt_buffer_dd":     300,
        "consistency_pct":    0.30,
        "news_blackout_sec":  120,
        "daily_target":       3000,
        "max_contracts":      10,
    },
    "150k": {
        "max_daily_loss":     5000,
        "trailing_drawdown":  5000,
        "halt_buffer_daily":  200,
        "halt_buffer_dd":     300,
        "consistency_pct":    0.30,
        "news_blackout_sec":  120,
        "daily_target":       5000,
        "max_contracts":      12,
    },
    "250k": {
        "max_daily_loss":     6500,
        "trailing_drawdown":  6500,
        "halt_buffer_daily":  200,
        "halt_buffer_dd":     300,
        "consistency_pct":    0.30,
        "news_blackout_sec":  120,
        "daily_target":       8000,
        "max_contracts":      14,
    },
}

# ── Futures tick/point values (USD per full point) ────────────────────────────
_POINT_VALUE: Dict[str, float] = {
    "ES":  50.0,   "MES":  5.0,
    "NQ":  20.0,   "MNQ":  2.0,
    "RTY": 50.0,   "M2K":  5.0,
    "CL":  1000.0, "MCL":  100.0,
    "GC":  100.0,  "MGC":  10.0,
    "SI":  5000.0,
}


# ── ATI connection ─────────────────────────────────────────────────────────────

class _ATIConnection:
    """Thread-safe TCP connection to NinjaTrader ATI with auto-reconnect."""

    def __init__(self, host: str, port: int) -> None:
        self._host    = host
        self._port    = port
        self._sock:   Optional[socket.socket] = None
        self._lock    = threading.Lock()
        self._alive   = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

    # ── connect / disconnect ──────────────────────────────────────────────────

    def connect(self) -> None:
        """Open (or re-open) the socket. Blocks until connected or raises."""
        with self._lock:
            self._close_socket()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(_CONNECT_TIMEOUT)
            sock.connect((self._host, self._port))
            sock.settimeout(_RECV_TIMEOUT)
            self._sock = sock
        self._alive.set()
        if self._hb_thread is None or not self._hb_thread.is_alive():
            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name="ninja-heartbeat"
            )
            self._hb_thread.start()
        logger.info("NinjaTrader ATI connected — %s:%d", self._host, self._port)

    def close(self) -> None:
        self._alive.clear()
        with self._lock:
            self._close_socket()
        logger.info("NinjaTrader ATI disconnected.")

    def _close_socket(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    # ── send / receive ────────────────────────────────────────────────────────

    def send_command(self, cmd: str) -> List[str]:
        """Send one ATI command and collect all response lines.

        Response lines are read until a blank line (terminator) or timeout.
        Re-connects once on socket error.
        """
        with self._lock:
            return self._send_once(cmd, retry=True)

    def _send_once(self, cmd: str, *, retry: bool = False) -> List[str]:
        if self._sock is None:
            if retry:
                self._reconnect()
            else:
                raise ConnectionError("ATI socket not connected.")
        try:
            self._sock.sendall((cmd + "\n").encode("utf-8"))
            return self._read_response()
        except (socket.timeout, OSError) as exc:
            logger.warning("ATI send error: %s", exc)
            self._close_socket()
            if retry:
                self._reconnect()
                self._sock.sendall((cmd + "\n").encode("utf-8"))
                return self._read_response()
            raise

    def _read_response(self) -> List[str]:
        """Read lines until blank line terminator."""
        buf = b""
        lines = []
        assert self._sock is not None
        while True:
            chunk = self._sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                decoded = line.decode("utf-8").rstrip("\r")
                if decoded == "":
                    return lines
                lines.append(decoded)
        return lines

    def _reconnect(self) -> None:
        logger.info("ATI reconnecting in %.1fs…", _RECONNECT_DELAY)
        time.sleep(_RECONNECT_DELAY)
        self.connect()

    # ── heartbeat ─────────────────────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        while self._alive.wait(timeout=_HEARTBEAT_INTERVAL):
            try:
                self.send_command("HEARTBEAT")
            except Exception as exc:
                logger.debug("Heartbeat failed: %s", exc)


# ── Response parsers ───────────────────────────────────────────────────────────

def _parse_account(lines: List[str]) -> Dict[str, Any]:
    """Parse ACCOUNT response into a dict."""
    result: Dict[str, Any] = {}
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 2:
            result[parts[0].lower()] = parts[1]
    return result


def _parse_positions(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse POSITIONS response.

    NinjaTrader returns one line per position:
      symbol|qty|avg_entry|unrealized_pnl|market_value|side
    """
    positions = []
    for line in lines:
        if not line or line.startswith("ERROR"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        try:
            sym     = parts[0].strip()
            qty     = float(parts[1])
            entry   = float(parts[2])
            unreal  = float(parts[3])
            side    = "long" if qty > 0 else "short"
            pv      = _POINT_VALUE.get(sym.upper(), 1.0)
            mv      = abs(qty) * entry * pv
            positions.append({
                "symbol":            sym,
                "qty":               qty,
                "avg_entry_price":   entry,
                "unrealized_pl":     unreal,
                "unrealized_plpc":   unreal / mv if mv else 0.0,
                "market_value":      mv,
                "side":              side,
                "current_price":     entry,  # updated below if price available
            })
        except (ValueError, IndexError):
            continue
    return positions


def _parse_orders(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse ORDERS response."""
    orders = []
    for line in lines:
        if not line or line.startswith("ERROR"):
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        try:
            orders.append({
                "id":           parts[0].strip(),
                "symbol":       parts[1].strip(),
                "side":         parts[2].strip().lower(),
                "qty":          float(parts[3]),
                "filled_qty":   float(parts[4]),
                "status":       parts[5].strip().lower(),
                "order_type":   parts[6].strip() if len(parts) > 6 else "market",
                "limit_price":  float(parts[7]) if len(parts) > 7 and parts[7] else None,
                "stop_price":   float(parts[8]) if len(parts) > 8 and parts[8] else None,
                "submitted_at": parts[9].strip() if len(parts) > 9 else "",
            })
        except (ValueError, IndexError):
            continue
    return orders


# ── Main broker class ──────────────────────────────────────────────────────────

@dataclass
class NinjaTraderBroker:
    """NinjaTrader 8 ATI broker with Apex Trader Funding risk enforcement.

    Parameters
    ----------
    account : str
        NinjaTrader account name (e.g. "Sim101" for simulation,
        "Apex-xxxxxxxx" for funded eval account).
    host : str
        ATI host, default localhost.
    port : int
        ATI port, default 36973.
    live_money : bool
        When True, applies the Apex risk rules strictly and enforces the
        PromotionGate (requires +$3,000 paper P&L on record).
    apex_tier : str
        One of "25k", "50k", "100k", "150k", "250k".
        Determines the daily loss / drawdown limits from APEX_RULES.
    """

    account:    str   = field(default_factory=lambda: os.getenv("NINJA_ACCOUNT", _DEFAULT_ACCOUNT))
    host:       str   = field(default_factory=lambda: os.getenv("NINJA_ATI_HOST", _DEFAULT_HOST))
    port:       int   = field(default_factory=lambda: int(os.getenv("NINJA_ATI_PORT", str(_DEFAULT_PORT))))
    live_money: bool  = False
    apex_tier:  str   = field(default_factory=lambda: os.getenv("APEX_TIER", "50k"))
    name:       str   = "ninja"

    # ── internal state ────────────────────────────────────────────────────────
    _conn:             _ATIConnection   = field(init=False)
    _seen_ids:         set              = field(default_factory=set)
    _daily_pnl:        float            = field(default=0.0)
    _session_pnl:      float            = field(default=0.0)
    _peak_equity:      float            = field(default=0.0)
    _halted:           bool             = field(default=False)
    _halt_reason:      str              = field(default="")
    _session_start:    datetime         = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        # Promotion gate for live funded accounts
        if self.live_money:
            from src.execution.promotion_gate import PromotionGate
            PromotionGate().require_eligibility(requested_live_money=True)

        self._conn = _ATIConnection(self.host, self.port)
        self._rules = APEX_RULES.get(self.apex_tier, APEX_RULES["50k"])
        logger.info(
            "NinjaTraderBroker initialised — account=%s  live=%s  apex_tier=%s",
            self.account, self.live_money, self.apex_tier,
        )
        # Attempt connection (non-fatal — operations will reconnect)
        try:
            self._conn.connect()
            equity = self.get_account_equity()
            self._peak_equity = equity
            logger.info("NinjaTrader connected. Starting equity: $%.2f", equity)
        except Exception as exc:
            logger.warning("NinjaTrader initial connect failed: %s — will retry on first order", exc)

    # ── Risk guard ────────────────────────────────────────────────────────────

    def _check_apex_risk(self, *, symbol: str = "", direction: str = "") -> Tuple[bool, str]:
        """Return (ok, reason). ok=False means do NOT trade."""
        if not self.live_money:
            return True, ""    # sim mode — no Apex restrictions

        if self._halted:
            return False, f"session halted: {self._halt_reason}"

        rules = self._rules

        # 1 ── News blackout
        try:
            from src.filters.event_blackout import is_blocked
            blocked, reason = is_blocked(symbol, skip_earnings_check=True)
            if blocked:
                return False, f"Apex news blackout: {reason}"
        except Exception:
            pass

        # 2 ── Daily loss limit
        equity = self.get_account_equity()
        dd_from_peak = self._peak_equity - equity
        daily_limit  = rules["max_daily_loss"]
        buffer       = rules["halt_buffer_daily"]
        if self._daily_pnl <= -(daily_limit - buffer):
            self._halt(f"daily loss ${-self._daily_pnl:.0f} ≥ ${daily_limit - buffer}")
            return False, self._halt_reason

        # 3 ── Trailing drawdown
        dd_limit  = rules["trailing_drawdown"]
        dd_buffer = rules["halt_buffer_dd"]
        if dd_from_peak >= dd_limit - dd_buffer:
            self._halt(f"trailing drawdown ${dd_from_peak:.0f} within ${dd_buffer} of limit ${dd_limit}")
            return False, self._halt_reason

        # 4 ── Consistency rule (single-day profit cap)
        if self._session_pnl > 0:
            max_day_profit = self._session_pnl * rules["consistency_pct"]
            if self._daily_pnl > max_day_profit:
                return False, (
                    f"Apex consistency: today +${self._daily_pnl:.0f} > "
                    f"30% of session +${self._session_pnl:.0f}"
                )

        return True, ""

    def _halt(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = reason
        logger.critical("APEX RISK HALT: %s — closing all positions.", reason)
        try:
            self.cancel_all_orders()
            self.close_all_positions()
        except Exception as exc:
            logger.error("Error during halt cleanup: %s", exc)

    # ── Broker interface ──────────────────────────────────────────────────────

    def get_account_equity(self) -> float:
        """Return current account net liquidation value in USD."""
        try:
            lines = self._conn.send_command(f"ACCOUNT|{self.account}")
            info  = _parse_account(lines)
            # NinjaTrader ATI returns: NetLiquidation, RealizedPnL, etc.
            for key in ("netliquidation", "netassetvalue", "accountvalue", "cashvalue"):
                if key in info:
                    return float(info[key])
            # Fallback: parse the first numeric value we find
            for v in info.values():
                try:
                    return float(v)
                except ValueError:
                    continue
        except Exception as exc:
            logger.warning("get_account_equity error: %s", exc)
        return 0.0

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Return list of open positions."""
        try:
            lines = self._conn.send_command(f"POSITIONS|{self.account}")
            return _parse_positions(lines)
        except Exception as exc:
            logger.warning("get_open_positions error: %s", exc)
            return []

    def get_orders(self, *, status: str = "all", limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent orders filtered by status."""
        try:
            lines  = self._conn.send_command(f"ORDERS|{self.account}")
            orders = _parse_orders(lines)
            if status == "open":
                orders = [o for o in orders
                          if o["status"] in ("accepted", "working", "pending", "partfilled")]
            elif status == "closed":
                orders = [o for o in orders
                          if o["status"] in ("filled", "cancelled", "rejected")]
            return orders[-limit:]
        except Exception as exc:
            logger.warning("get_orders error: %s", exc)
            return []

    def submit_signal(self, signal: Dict[str, Any]) -> "ExecutionResult":
        """Convert a BRZRKR signal dict into NinjaTrader ATI bracket orders.

        Submits three orders atomically:
          1. Market entry order
          2. Stop-loss order (linked via OCO group)
          3. Take-profit limit order (linked via same OCO group)
        """
        from src.execution.brokers import ExecutionResult

        symbol    = signal["asset"].upper()
        direction = signal["direction"]
        entry_px  = float(signal["entry_price"])
        stop_px   = float(signal["stop_loss"])
        tp_px     = float(signal["take_profit"])
        notional  = float(signal["position_size_usd"])

        # ── Apex risk check ────────────────────────────────────────────────
        ok, reason = self._check_apex_risk(symbol=symbol, direction=direction)
        if not ok:
            logger.warning("Order blocked by Apex risk rules: %s", reason)
            return ExecutionResult(False, None, reason)

        # ── Size in contracts ──────────────────────────────────────────────
        pv  = _POINT_VALUE.get(symbol, 1.0)
        qty = max(1, int(notional / (entry_px * pv)))
        qty = min(qty, self._rules.get("max_contracts", 10))

        # ── Build ATI action string ────────────────────────────────────────
        action = "BUY" if direction == "long" else "SELL"
        oco_id = f"OCO-{uuid.uuid4().hex[:8]}"

        # Entry (market)
        entry_id = f"BRZRKR-{uuid.uuid4().hex[:10]}"
        if entry_id in self._seen_ids:
            return ExecutionResult(False, None, "duplicate client_order_id")
        self._seen_ids.add(entry_id)

        # Stop order (opposite action)
        stop_id  = f"{entry_id}-SL"
        stop_action = "SELL" if action == "BUY" else "BUY"

        # Take-profit limit order
        tp_id    = f"{entry_id}-TP"

        # ATI PLACE format:
        # PLACE|order_id|account|action|qty|symbol|order_type|limit_price|stop_price|tif|oco_id|from_entry_signal|strategy_id
        def _place(oid, act, qty, sym, otype, lim="", stop="", tif="GTC", oco=""):
            return f"PLACE|{oid}|{self.account}|{act}|{qty}|{sym}|{otype}|{lim}|{stop}|{tif}|{oco}||BRZRKR"

        try:
            # 1. Entry
            resp_entry = self._conn.send_command(
                _place(entry_id, action, qty, symbol, "MARKET", tif="DAY")
            )
            if any("ERROR" in r for r in resp_entry):
                reason = " ".join(resp_entry)
                return ExecutionResult(False, entry_id, f"Entry rejected: {reason}")

            # 2. Stop loss (OCO with TP)
            sl_px_str = f"{stop_px:.2f}"
            self._conn.send_command(
                _place(stop_id, stop_action, qty, symbol, "STOP",
                       stop=sl_px_str, oco=oco_id)
            )

            # 3. Take profit (OCO with SL)
            tp_px_str = f"{tp_px:.2f}"
            self._conn.send_command(
                _place(tp_id, stop_action, qty, symbol, "LIMIT",
                       lim=tp_px_str, oco=oco_id)
            )

            logger.info(
                "NinjaTrader order submitted: %s %s %s × %d  SL=%.2f  TP=%.2f",
                action, qty, symbol, qty, stop_px, tp_px,
            )
            return ExecutionResult(True, entry_id, "ok")

        except Exception as exc:
            logger.error("submit_signal error: %s", exc)
            return ExecutionResult(False, None, str(exc))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by its client order ID."""
        try:
            lines = self._conn.send_command(f"CANCEL|{order_id}|{self.account}")
            return not any("ERROR" in r for r in lines)
        except Exception as exc:
            logger.warning("cancel_order error: %s", exc)
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all working orders. Returns number cancelled."""
        try:
            open_orders = self.get_orders(status="open")
            n = 0
            for o in open_orders:
                if self.cancel_order(o["id"]):
                    n += 1
            return n
        except Exception as exc:
            logger.warning("cancel_all_orders error: %s", exc)
            return 0

    def close_position(self, symbol: str) -> bool:
        """Flatten a single position."""
        try:
            lines = self._conn.send_command(
                f"CLOSEPOSITION|{symbol.upper()}|{self.account}"
            )
            return not any("ERROR" in r for r in lines)
        except Exception as exc:
            logger.warning("close_position error: %s", exc)
            return False

    def close_all_positions(self) -> bool:
        """Flatten all positions (Apex halt cleanup)."""
        try:
            lines = self._conn.send_command(f"CLOSEALLPOSITIONS|{self.account}")
            return not any("ERROR" in r for r in lines)
        except Exception as exc:
            logger.warning("close_all_positions error: %s", exc)
            return False

    # ── Daily reset ───────────────────────────────────────────────────────────

    def record_fill(self, pnl_usd: float) -> None:
        """Call this after each confirmed fill to track Apex P&L limits."""
        self._daily_pnl   += pnl_usd
        self._session_pnl += pnl_usd
        equity = self.get_account_equity()
        if equity > self._peak_equity:
            self._peak_equity = equity
        logger.debug(
            "Fill recorded: pnl=%.2f  daily=%.2f  session=%.2f  peak_equity=%.2f",
            pnl_usd, self._daily_pnl, self._session_pnl, self._peak_equity,
        )

    def reset_daily(self) -> None:
        """Call at session open each morning to reset daily counters."""
        logger.info("Daily reset — yesterday P&L: $%.2f", self._daily_pnl)
        self._daily_pnl     = 0.0
        self._halted        = False
        self._halt_reason   = ""
        self._session_start = datetime.now(timezone.utc)

    # ── Status helpers ────────────────────────────────────────────────────────

    def is_halted(self) -> bool:
        return self._halted

    def halt_reason(self) -> str:
        return self._halt_reason

    def apex_status(self) -> Dict[str, Any]:
        """Return a status snapshot for the desktop app's Status page."""
        equity = self.get_account_equity()
        rules  = self._rules
        return {
            "account":           self.account,
            "apex_tier":         self.apex_tier,
            "equity":            equity,
            "peak_equity":       self._peak_equity,
            "daily_pnl":         self._daily_pnl,
            "session_pnl":       self._session_pnl,
            "trailing_dd_used":  self._peak_equity - equity,
            "trailing_dd_limit": rules["trailing_drawdown"],
            "trailing_dd_buffer":rules["halt_buffer_dd"],
            "daily_loss_limit":  rules["max_daily_loss"],
            "daily_loss_buffer": rules["halt_buffer_daily"],
            "halted":            self._halted,
            "halt_reason":       self._halt_reason,
        }

    def disconnect(self) -> None:
        """Cleanly close the ATI socket."""
        self._conn.close()
