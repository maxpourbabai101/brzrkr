"""trading_enhancer dashboard — Streamlit single-file app.

Run:
    streamlit run dashboard.py

Three tabs:
    1. System Health   — connection status for every component,
                         recent log activity, recent signals, equity snapshot.
    2. Active Trades   — open positions table + chart, open orders table,
                         today's P&L.
    3. Trade Controls  — form for new orders, per-position close buttons,
                         cancel-all and per-order cancel buttons.

Read-only state polls every refresh; mutating actions (place/cancel/close)
fire on button click. Use the sidebar's "Auto-refresh" toggle for hands-off
monitoring.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.broker import AlpacaExecutor
from src.signals.signal_generator import generate_signal

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="trading_enhancer",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources / data
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_executor(live_money: bool = False) -> AlpacaExecutor:
    return AlpacaExecutor(live_money=live_money)


@st.cache_data(ttl=10, show_spinner=False)
def _broker_snapshot(_executor_id: int) -> Dict[str, Any]:
    """Pulls all read-only broker state in one batch."""
    ex = _get_executor()
    return {
        "equity": ex.get_account_equity(),
        "positions": ex.get_open_positions(),
        "orders": ex.get_orders(status="all", limit=50),
        "paper": ex._paper,
    }


def _bust_broker_cache() -> None:
    _broker_snapshot.clear()


def _read_recent_signals(limit: int = 12) -> List[Dict[str, Any]]:
    d = Path("data/signals")
    if not d.exists():
        return []
    paths = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in paths:
        try:
            data = json.loads(p.read_text())
            data["_file"] = p.name
            data["_mtime"] = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            out.append(data)
        except Exception:  # noqa: BLE001
            continue
    return out


def _read_log_tail(n: int = 40) -> str:
    log = Path("trading_enhancer.log")
    if not log.exists():
        return "(no log file yet)"
    try:
        lines = log.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:  # noqa: BLE001
        return f"(log read failed: {exc})"


def _check_env_keys() -> Dict[str, bool]:
    """True iff the env var is present and non-empty."""
    keys = [
        "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
        "FRED_API_KEY", "FINNHUB_API_KEY",
        "NEWSAPI_KEY", "ALPHA_VANTAGE_API_KEY",
        "TRADIER_API_KEY", "POLYGON_API_KEY",
    ]
    return {k: bool(os.getenv(k)) for k in keys}


AGENT_PID_FILE = Path("agent.pid")
AGENT_STOP_FILE = Path("AGENT_STOP")
AGENT_LOG_FILE = Path("agent.out")


def _agent_pid() -> int | None:
    if not AGENT_PID_FILE.exists():
        return None
    try:
        return int(AGENT_PID_FILE.read_text().strip())
    except Exception:  # noqa: BLE001
        return None


def _agent_running() -> bool:
    pid = _agent_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def _stop_file_present() -> bool:
    return AGENT_STOP_FILE.exists()


def _start_agent(*, dry_run: bool, tick_seconds: int, source: str) -> tuple[bool, str]:
    """Spawn agent.py as a detached subprocess. Returns (ok, message)."""
    if _agent_running():
        return False, f"Agent already running (PID {_agent_pid()})."

    # Clear any stale stop file so the new agent doesn't exit immediately.
    if AGENT_STOP_FILE.exists():
        try:
            AGENT_STOP_FILE.unlink()
        except Exception:  # noqa: BLE001
            pass

    cmd = [sys.executable, "agent.py",
           "--dry-run" if dry_run else "--execute",
           "--tick-seconds", str(tick_seconds),
           "--source", source]

    log = open(AGENT_LOG_FILE, "ab")
    proc = subprocess.Popen(
        cmd,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,    # detach from this process group
        cwd=str(ROOT),
    )
    AGENT_PID_FILE.write_text(str(proc.pid))
    return True, f"Agent started (PID {proc.pid}). Logs → `agent.out`."


def _stop_agent() -> tuple[bool, str]:
    if not _agent_running():
        # Clean up stale pid file.
        if AGENT_PID_FILE.exists():
            AGENT_PID_FILE.unlink()
        return False, "Agent is not running."
    AGENT_STOP_FILE.touch()
    return True, ("Stop file written. Agent will exit within ~2s of its "
                  "next stop-check. Polling…")


def _kill_agent() -> tuple[bool, str]:
    """Force-kill the agent process (SIGTERM). Last resort."""
    pid = _agent_pid()
    if pid is None:
        return False, "No agent.pid file."
    try:
        os.kill(pid, 15)   # SIGTERM
        return True, f"Sent SIGTERM to PID {pid}."
    except Exception as exc:  # noqa: BLE001
        return False, f"Kill failed: {exc}"


def _read_agent_log_tail(n: int = 30) -> str:
    if not AGENT_LOG_FILE.exists():
        return "(no agent.out yet — agent hasn't been started from the dashboard)"
    try:
        return "\n".join(AGENT_LOG_FILE.read_text(errors="replace").splitlines()[-n:])
    except Exception as exc:  # noqa: BLE001
        return f"(read failed: {exc})"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 trading_enhancer")
    st.caption("Autonomous trading dashboard")
    st.divider()

    auto = st.checkbox("🔄 Auto-refresh (10s)", value=False,
                       help="Re-poll the broker every 10 seconds.")
    if st.button("Refresh now", use_container_width=True):
        _bust_broker_cache()
        st.rerun()

    st.divider()
    st.subheader("Quick links")
    st.markdown(
        "- [Alpaca paper dashboard](https://app.alpaca.markets/paper/dashboard/overview)\n"
        "- [Alpaca live dashboard](https://app.alpaca.markets/dashboard/overview)\n"
        "- [Project README](./README.md)"
    )


# ---------------------------------------------------------------------------
# Try to connect to the broker. If creds are missing, render a clean error.
# ---------------------------------------------------------------------------
try:
    snap = _broker_snapshot(id(_get_executor()))
    broker_ok = True
    broker_err: str | None = None
except Exception as exc:  # noqa: BLE001
    broker_ok = False
    broker_err = str(exc)
    snap = {"equity": 0.0, "positions": [], "orders": [], "paper": True}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_health, tab_trades, tab_controls = st.tabs(
    ["🩺 System Health", "💹 Active Trades", "🛒 Trade Controls"]
)


# ===========================================================================
# Tab 1 — System Health
# ===========================================================================
with tab_health:
    st.header("System Health")

    c1, c2, c3, c4 = st.columns(4)

    # Broker
    if broker_ok:
        endpoint = "Paper" if snap["paper"] else "🔴 LIVE"
        c1.metric("Broker", f"✅ {endpoint}", f"${snap['equity']:,.2f}")
    else:
        c1.metric("Broker", "❌ Down", (broker_err or "")[:40])

    # Agent (full controls live in the panel below).
    if _agent_running():
        c2.metric("Agent", "✅ Running", f"PID {_agent_pid()}")
    elif _stop_file_present():
        c2.metric("Agent", "⏸ Stop set")
    else:
        c2.metric("Agent", "⚫ Stopped")

    # Positions / orders summary
    c3.metric("Open positions", len(snap["positions"]))
    open_orders = [o for o in snap["orders"]
                   if o["status"] in ("new", "accepted", "pending_new",
                                      "partially_filled")]
    c4.metric("Open orders", len(open_orders))

    st.divider()

    # ---- Agent control panel ---------------------------------------------
    st.subheader("Agent control")
    pid = _agent_pid()
    running = _agent_running()
    cstat, cstart, cstop, ckill = st.columns([2, 1, 1, 1])

    if running:
        cstat.success(f"✅ Agent running (PID {pid})")
    elif _stop_file_present():
        cstat.warning("⏸ Stop file present — agent halting / stopped")
    else:
        cstat.info("⚫ Agent not running")

    with cstart.popover("▶️ Start", use_container_width=True,
                        disabled=running):
        st.caption("Spawn agent.py as a background process.")
        s_dry = st.checkbox("Dry run (no orders)", value=True, key="start_dry")
        s_tick = st.number_input("Tick seconds", min_value=10, value=60,
                                  step=10, key="start_tick")
        s_source = st.selectbox("Data source", ["api", "scraper", "both"],
                                index=0, key="start_source")
        if st.button("Start now", key="start_btn", type="primary"):
            ok, msg = _start_agent(dry_run=s_dry, tick_seconds=int(s_tick),
                                    source=s_source)
            (st.success if ok else st.error)(msg)
            time.sleep(0.4)
            st.rerun()

    if cstop.button("⏸ Stop", use_container_width=True, disabled=not running):
        ok, msg = _stop_agent()
        (st.success if ok else st.warning)(msg)
        time.sleep(0.4)
        st.rerun()

    if ckill.button("🛑 Kill", use_container_width=True, disabled=not running,
                    help="Force SIGTERM. Use only if Stop hangs."):
        ok, msg = _kill_agent()
        (st.success if ok else st.error)(msg)
        time.sleep(0.4)
        st.rerun()

    with st.expander("Agent log tail (`agent.out`)", expanded=False):
        st.code(_read_agent_log_tail(40), language="text")

    st.divider()

    # API key matrix
    st.subheader("API keys")
    keys = _check_env_keys()
    cols = st.columns(4)
    for i, (k, present) in enumerate(keys.items()):
        cols[i % 4].markdown(
            f"{'✅' if present else '⬜'} `{k.replace('_API_KEY','').replace('_KEY','').replace('_SECRET','SECRET')}`"
        )

    st.caption(
        "Missing keys aren't fatal — every loader gracefully degrades. "
        "Alpaca's two keys are the only ones required to trade."
    )

    st.divider()

    # Equity / P&L mini chart from positions (simple sum)
    if snap["positions"]:
        df_pos = pd.DataFrame(snap["positions"])
        total_pnl = float(df_pos["unrealized_pl"].sum())
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Account equity", f"${snap['equity']:,.2f}")
        col_b.metric("Unrealized P&L", f"${total_pnl:+,.2f}")
        col_c.metric("Symbols held", df_pos["symbol"].nunique())
    else:
        st.info("No positions held — nothing to chart yet.")

    st.divider()

    # Recent signals
    st.subheader("Recent signal records")
    signals = _read_recent_signals(limit=12)
    if signals:
        rows = []
        for s in signals:
            rows.append({
                "time": s.get("_mtime"),
                "asset": s.get("asset"),
                "direction": s.get("direction"),
                "entry": s.get("entry_price"),
                "stop": s.get("stop_loss"),
                "tp": s.get("take_profit"),
                "size_$": s.get("position_size_usd"),
                "confidence": s.get("confidence"),
                "file": s.get("_file"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No signal files in `data/signals/` yet.")

    # Log tail
    st.subheader("Recent log activity")
    st.code(_read_log_tail(40), language="text")


# ===========================================================================
# Tab 2 — Active Trades
# ===========================================================================
with tab_trades:
    st.header("Active Trades")

    if not broker_ok:
        st.error(f"Broker offline: {broker_err}")
    else:
        # Top metrics
        equity = snap["equity"]
        positions = snap["positions"]
        total_mv = sum(p["market_value"] for p in positions) if positions else 0.0
        total_pnl = sum(p["unrealized_pl"] for p in positions) if positions else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Account equity", f"${equity:,.2f}")
        m2.metric("Position MV", f"${total_mv:,.2f}")
        m3.metric("Unrealized P&L", f"${total_pnl:+,.2f}",
                  delta=f"{(total_pnl/equity*100):+.2f}%" if equity > 0 else None)
        m4.metric("# positions", len(positions))

        st.divider()

        # Positions table + bar chart
        st.subheader("Open positions")
        if positions:
            dfp = pd.DataFrame(positions)
            dfp_display = dfp.assign(
                pnl=lambda d: d["unrealized_pl"].map(lambda x: f"${x:+,.2f}"),
                pnl_pct=lambda d: (d["unrealized_plpc"] * 100).map(lambda x: f"{x:+.2f}%"),
            )[["symbol", "side", "qty", "avg_entry_price",
               "current_price", "market_value", "pnl", "pnl_pct"]]
            st.dataframe(dfp_display, hide_index=True, use_container_width=True)

            # Color-coded P&L bar chart.
            fig = px.bar(
                dfp, x="symbol", y="unrealized_pl",
                color=dfp["unrealized_pl"].map(lambda x: "Gain" if x >= 0 else "Loss"),
                color_discrete_map={"Gain": "#16a34a", "Loss": "#dc2626"},
                title="Unrealized P&L by position",
            )
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions.")

        st.divider()

        # Orders table
        st.subheader("Recent orders")
        if snap["orders"]:
            dfo = pd.DataFrame(snap["orders"])[
                ["submitted_at", "symbol", "side", "qty", "filled_qty",
                 "order_type", "status", "id"]
            ]
            st.dataframe(dfo, hide_index=True, use_container_width=True)
        else:
            st.info("No orders.")


# ===========================================================================
# Tab 3 — Trade Controls
# ===========================================================================
with tab_controls:
    st.header("Trade Controls")
    if not broker_ok:
        st.error(f"Broker offline — controls disabled. ({broker_err})")
    else:
        # ---- New order form ----------------------------------------------
        st.subheader("Place a new order")
        with st.form("new_order"):
            c1, c2, c3 = st.columns([1, 1, 1])
            symbol = c1.text_input("Symbol", value="SPY").strip().upper()
            direction = c2.radio("Side", ["long", "short"], horizontal=True)
            notional = c3.number_input(
                "Notional ($)", min_value=1.0, value=1000.0, step=100.0
            )
            c4, c5 = st.columns(2)
            stop_pct = c4.slider("Stop loss (%)", 0.1, 10.0, 1.0, 0.1) / 100.0
            tp_pct = c5.slider("Take profit (%)", 0.1, 20.0, 2.0, 0.1) / 100.0
            confirm = st.checkbox("I have reviewed the parameters", value=False)
            submit = st.form_submit_button("Place order", type="primary",
                                           use_container_width=True)

        if submit:
            if not confirm:
                st.warning("Tick the 'reviewed' box to enable submission.")
            elif not symbol:
                st.warning("Symbol required.")
            else:
                ex = _get_executor()
                # Pull last trade price from Alpaca's market data for entry.
                try:
                    import requests
                    r = requests.get(
                        f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                        headers={
                            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
                            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
                        },
                        timeout=10,
                    )
                    r.raise_for_status()
                    entry = float((r.json().get("trade") or {}).get("p", 0))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Quote lookup failed: {exc}")
                    entry = 0.0

                if entry > 0:
                    if direction == "long":
                        stop = entry * (1 - stop_pct)
                        tp = entry * (1 + tp_pct)
                    else:
                        stop = entry * (1 + stop_pct)
                        tp = entry * (1 - tp_pct)
                    sig = {
                        "asset": symbol,
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "direction": direction,
                        "entry_price": round(entry, 2),
                        "stop_loss": round(stop, 2),
                        "take_profit": round(tp, 2),
                        "position_size_usd": float(notional),
                        "expected_return_pct": tp_pct,
                        "iv_change_pct": 0.0,
                        "confidence": 1.0,
                        "risk_flags": {"manual_dashboard": True},
                    }
                    with st.expander("Order payload", expanded=False):
                        st.json(sig)
                    result = ex.submit_signal(sig)
                    if result.submitted:
                        st.success(f"✅ Order submitted — id `{result.order_id}`")
                        _bust_broker_cache()
                    else:
                        st.error(f"❌ Order rejected: {result.reason}")

        st.divider()

        # ---- Position closure --------------------------------------------
        st.subheader("Close positions")
        if not snap["positions"]:
            st.caption("No open positions.")
        for p in snap["positions"]:
            cols = st.columns([3, 2, 1])
            cols[0].write(f"**{p['symbol']}** — qty {p['qty']}  ·  side: {p['side']}")
            cols[1].write(
                f"MV ${p['market_value']:,.2f}  ·  PnL ${p['unrealized_pl']:+,.2f}"
            )
            if cols[2].button("Close", key=f"close_{p['symbol']}",
                              use_container_width=True):
                ex = _get_executor()
                if ex.close_position(p["symbol"]):
                    st.success(f"Close requested for {p['symbol']}")
                else:
                    st.error(f"Close failed for {p['symbol']}")
                _bust_broker_cache()
                time.sleep(0.5)
                st.rerun()

        st.divider()

        # ---- Order cancellation ------------------------------------------
        st.subheader("Cancel orders")
        open_ords = [o for o in snap["orders"]
                     if o["status"] in ("new", "accepted", "pending_new",
                                        "partially_filled")]
        if not open_ords:
            st.caption("No open orders.")

        for o in open_ords:
            cols = st.columns([3, 2, 1])
            cols[0].write(
                f"**{o['symbol']}** {o['side']} qty {o['qty']} "
                f"({o['order_type']})"
            )
            cols[1].write(f"status: `{o['status']}`")
            if cols[2].button("Cancel", key=f"cancel_{o['id']}",
                              use_container_width=True):
                ex = _get_executor()
                if ex.cancel_order(o["id"]):
                    st.success(f"Cancel requested for {o['symbol']}")
                else:
                    st.error(f"Cancel failed for {o['id']}")
                _bust_broker_cache()
                time.sleep(0.5)
                st.rerun()

        if open_ords:
            st.divider()
            if st.button("⚠️ Cancel ALL open orders", type="secondary",
                         use_container_width=True):
                ex = _get_executor()
                n = ex.cancel_all_orders()
                st.success(f"Cancelled {n} order(s).")
                _bust_broker_cache()
                time.sleep(0.5)
                st.rerun()


# ---------------------------------------------------------------------------
# Auto-refresh trigger (must be at the bottom so all widgets exist first)
# ---------------------------------------------------------------------------
if auto:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=10_000, limit=None, key="dash_autorefresh")
    except ImportError:
        # Fallback: simple sleep + rerun. Not as clean but works.
        time.sleep(10)
        _bust_broker_cache()
        st.rerun()
