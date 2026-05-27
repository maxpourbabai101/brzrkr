"""trading_enhancer — native macOS desktop app.

CustomTkinter-based UI you can drop on the Dock. Three sections in a
fixed sidebar:

    🩺 Health      — broker state, agent process, log tail, API matrix
    💹 Positions   — open positions + recent orders (auto-refreshing)
    🛒 Controls    — new order form, per-row close/cancel buttons

Launch directly:
    python desktop_app.py

Drop on the Dock:
    1. Use launcher.command (drag to Dock) — opens a tiny terminal then
       runs this app.
    2. Or build a true .app bundle:  ./build_macapp.sh
       The resulting `dist/trading_enhancer.app` is dock-droppable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

import customtkinter as ctk
from tkinter import font as tkfont
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.broker import AlpacaExecutor

# ---------------------------------------------------------------------------
# Theme — Bloomberg/Refinitiv-inspired dark palette
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

COLORS = {
    "bg":           "#0b0f1a",     # window background
    "panel":        "#131826",     # card backgrounds
    "panel_hi":     "#1c2436",     # hover / elevated
    "border":       "#252e44",
    "accent":       "#3b82f6",     # primary blue
    "accent_hi":    "#60a5fa",
    "text":         "#e2e8f0",     # primary text
    "text_muted":   "#94a3b8",     # secondary text
    "text_dim":     "#64748b",     # tertiary
    "green":        "#10b981",
    "green_dim":    "#065f46",
    "red":          "#ef4444",
    "red_dim":      "#7f1d1d",
    "yellow":       "#f59e0b",
    "row_alt":      "#0f1422",
}

MONO_FONT = ("SF Mono", "Menlo", "Monaco", "Consolas")
SANS_FONT = ("SF Pro Text", "Helvetica Neue", "Helvetica", "Arial")

AGENT_PID_FILE = ROOT / "agent.pid"
AGENT_STOP_FILE = ROOT / "AGENT_STOP"
AGENT_LOG_FILE = ROOT / "agent.out"
APP_LOG_FILE = ROOT / "trading_enhancer.log"
SIGNALS_DIR = ROOT / "data" / "signals"


# ---------------------------------------------------------------------------
# Background polling thread — keeps the UI from blocking on HTTP
# ---------------------------------------------------------------------------
class BrokerPoller(threading.Thread):
    def __init__(self, queue: Queue, interval: float = 8.0) -> None:
        super().__init__(daemon=True)
        self.queue = queue
        self.interval = interval
        self._stop = threading.Event()
        self._executor: Optional[AlpacaExecutor] = None

    def stop(self) -> None:
        self._stop.set()

    def trigger(self) -> None:
        """Force an immediate poll on next loop iteration."""
        self._fast_next = True

    def run(self) -> None:
        self._fast_next = False
        while not self._stop.is_set():
            self.queue.put(self._snapshot())
            for _ in range(int(self.interval * 10)):
                if self._stop.is_set() or self._fast_next:
                    self._fast_next = False
                    break
                time.sleep(0.1)

    def _snapshot(self) -> Dict[str, Any]:
        try:
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
            }


# ---------------------------------------------------------------------------
# Reusable UI widgets
# ---------------------------------------------------------------------------
class Card(ctk.CTkFrame):
    def __init__(self, parent, **kw):
        kw.setdefault("fg_color", COLORS["panel"])
        kw.setdefault("border_color", COLORS["border"])
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 8)
        super().__init__(parent, **kw)


class Metric(ctk.CTkFrame):
    """Big number + label card."""

    def __init__(self, parent, label: str, value: str = "—",
                 sub: str = "", color: str | None = None):
        super().__init__(parent, fg_color=COLORS["panel"],
                         border_color=COLORS["border"], border_width=1,
                         corner_radius=8)
        self.grid_columnconfigure(0, weight=1)
        self.label = ctk.CTkLabel(
            self, text=label.upper(),
            text_color=COLORS["text_muted"],
            font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold"),
            anchor="w",
        )
        self.label.grid(row=0, column=0, padx=16, pady=(14, 2), sticky="ew")

        self.value = ctk.CTkLabel(
            self, text=value,
            text_color=color or COLORS["text"],
            font=ctk.CTkFont(family=MONO_FONT[0], size=26, weight="bold"),
            anchor="w",
        )
        self.value.grid(row=1, column=0, padx=16, pady=0, sticky="ew")

        self.sub = ctk.CTkLabel(
            self, text=sub,
            text_color=COLORS["text_dim"],
            font=ctk.CTkFont(family=SANS_FONT[0], size=11),
            anchor="w",
        )
        self.sub.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="ew")

    def set(self, value: str, *, sub: str = "", color: str | None = None) -> None:
        self.value.configure(text=value, text_color=color or COLORS["text"])
        self.sub.configure(text=sub)


class StatusPill(ctk.CTkFrame):
    def __init__(self, parent, text: str = "", state: str = "neutral"):
        super().__init__(parent, fg_color="transparent")
        self._dot = ctk.CTkLabel(self, text="●", font=ctk.CTkFont(size=14),
                                  text_color=COLORS["text_dim"])
        self._dot.pack(side="left", padx=(0, 6))
        self._lbl = ctk.CTkLabel(self, text=text,
                                  text_color=COLORS["text"],
                                  font=ctk.CTkFont(family=SANS_FONT[0],
                                                   size=12, weight="bold"))
        self._lbl.pack(side="left")
        self.set(text, state)

    def set(self, text: str, state: str = "neutral") -> None:
        colors = {"ok": COLORS["green"], "warn": COLORS["yellow"],
                  "err": COLORS["red"], "neutral": COLORS["text_dim"]}
        self._dot.configure(text_color=colors.get(state, COLORS["text_dim"]))
        self._lbl.configure(text=text)


# ---------------------------------------------------------------------------
# Page: Health
# ---------------------------------------------------------------------------
class HealthPage(ctk.CTkFrame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, fg_color=COLORS["bg"])
        self.app = app

        for i in range(4):
            self.grid_columnconfigure(i, weight=1)

        # Top metrics
        self.m_broker = Metric(self, "Broker", "Connecting…", "—")
        self.m_equity = Metric(self, "Account Equity", "$ —")
        self.m_positions = Metric(self, "Open Positions", "0")
        self.m_orders = Metric(self, "Open Orders", "0")
        self.m_broker.grid(row=0, column=0, padx=(0, 8), pady=(0, 12), sticky="nsew")
        self.m_equity.grid(row=0, column=1, padx=8, pady=(0, 12), sticky="nsew")
        self.m_positions.grid(row=0, column=2, padx=8, pady=(0, 12), sticky="nsew")
        self.m_orders.grid(row=0, column=3, padx=(8, 0), pady=(0, 12), sticky="nsew")

        # Agent control panel
        agent = Card(self)
        agent.grid(row=1, column=0, columnspan=4, sticky="nsew", pady=(0, 12))
        agent.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(agent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="AGENT", text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, sticky="w")
        self.agent_status = StatusPill(header, "Checking…", "neutral")
        self.agent_status.grid(row=0, column=1, sticky="e")

        btns = ctk.CTkFrame(agent, fg_color="transparent")
        btns.grid(row=1, column=0, sticky="ew", padx=16, pady=(6, 14))
        for i in range(5):
            btns.grid_columnconfigure(i, weight=1, uniform="agent")

        self.var_dry = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(btns, text="Dry run (no orders)", variable=self.var_dry,
                        fg_color=COLORS["accent"], hover_color=COLORS["accent_hi"]
                        ).grid(row=0, column=0, sticky="w")

        self.var_tick = ctk.StringVar(value="60")
        ctk.CTkEntry(btns, textvariable=self.var_tick, width=80,
                     placeholder_text="tick s").grid(row=0, column=1, sticky="w", padx=8)

        self.var_source = ctk.StringVar(value="api")
        ctk.CTkOptionMenu(btns, values=["api", "scraper", "both"],
                          variable=self.var_source, width=110,
                          fg_color=COLORS["panel_hi"],
                          button_color=COLORS["accent"]
                          ).grid(row=0, column=2, sticky="w")

        self.btn_start = ctk.CTkButton(btns, text="▶  Start Agent",
                                       fg_color=COLORS["accent"],
                                       hover_color=COLORS["accent_hi"],
                                       command=self.start_agent)
        self.btn_start.grid(row=0, column=3, sticky="ew", padx=(8, 4))
        self.btn_stop = ctk.CTkButton(btns, text="■  Stop",
                                      fg_color=COLORS["red_dim"],
                                      hover_color=COLORS["red"],
                                      command=self.stop_agent)
        self.btn_stop.grid(row=0, column=4, sticky="ew", padx=(4, 0))

        # API matrix
        api_card = Card(self)
        api_card.grid(row=2, column=0, columnspan=2, sticky="nsew",
                      padx=(0, 6), pady=(0, 12))
        api_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(api_card, text="API KEYS", text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")
        self.api_grid = ctk.CTkFrame(api_card, fg_color="transparent")
        self.api_grid.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))
        for i in range(2):
            self.api_grid.grid_columnconfigure(i, weight=1)
        self._api_labels: Dict[str, ctk.CTkLabel] = {}
        keys = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY",
                "FRED_API_KEY", "FINNHUB_API_KEY",
                "NEWSAPI_KEY", "ALPHA_VANTAGE_API_KEY",
                "TRADIER_API_KEY", "POLYGON_API_KEY"]
        for i, k in enumerate(keys):
            lbl = ctk.CTkLabel(
                self.api_grid, text=f"○  {k}",
                anchor="w",
                font=ctk.CTkFont(family=MONO_FONT[0], size=11),
                text_color=COLORS["text_dim"],
            )
            lbl.grid(row=i // 2, column=i % 2, padx=4, pady=2, sticky="ew")
            self._api_labels[k] = lbl

        # Log tail
        log_card = Card(self)
        log_card.grid(row=2, column=2, columnspan=2, sticky="nsew",
                      padx=(6, 0), pady=(0, 12))
        log_card.grid_rowconfigure(1, weight=1)
        log_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(log_card, text="LOG TAIL", text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")
        self.log_view = ctk.CTkTextbox(
            log_card, fg_color=COLORS["bg"], text_color=COLORS["text_muted"],
            border_color=COLORS["border"], border_width=1,
            font=ctk.CTkFont(family=MONO_FONT[0], size=10),
            wrap="none", height=240,
        )
        self.log_view.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        # Recent signals
        sig_card = Card(self)
        sig_card.grid(row=3, column=0, columnspan=4, sticky="nsew")
        sig_card.grid_columnconfigure(0, weight=1)
        sig_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(sig_card, text="RECENT SIGNALS",
                     text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")
        self.signals_view = ctk.CTkTextbox(
            sig_card, fg_color=COLORS["bg"], text_color=COLORS["text"],
            border_color=COLORS["border"], border_width=1,
            font=ctk.CTkFont(family=MONO_FONT[0], size=11),
            wrap="none", height=160,
        )
        self.signals_view.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

    # ---- callbacks --------------------------------------------------------
    def start_agent(self) -> None:
        if self._agent_running():
            self.app.toast("Agent already running.")
            return
        if AGENT_STOP_FILE.exists():
            try: AGENT_STOP_FILE.unlink()
            except Exception: pass
        try:
            tick = max(10, int(self.var_tick.get() or "60"))
        except ValueError:
            tick = 60
        cmd = [sys.executable, str(ROOT / "agent.py"),
               "--dry-run" if self.var_dry.get() else "--execute",
               "--tick-seconds", str(tick),
               "--source", self.var_source.get()]
        log = open(AGENT_LOG_FILE, "ab")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(ROOT))
        AGENT_PID_FILE.write_text(str(proc.pid))
        self.app.toast(f"Agent started (PID {proc.pid})")

    def stop_agent(self) -> None:
        if not self._agent_running():
            self.app.toast("Agent isn't running.")
            return
        AGENT_STOP_FILE.touch()
        self.app.toast("Stop file written — agent will exit within ~2s.")

    @staticmethod
    def _agent_running() -> bool:
        if not AGENT_PID_FILE.exists():
            return False
        try:
            pid = int(AGENT_PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    # ---- update from poller ----------------------------------------------
    def update_from(self, snap: Dict[str, Any]) -> None:
        if snap.get("ok"):
            endpoint = "Paper" if snap["paper"] else "🔴 LIVE"
            self.m_broker.set(f"●  {endpoint}", sub="Connected",
                              color=COLORS["green"])
            self.m_equity.set(f"${snap['equity']:,.2f}")
            self.m_positions.set(str(len(snap["positions"])))
            open_orders = [o for o in snap["orders"]
                           if o["status"] in ("new", "accepted", "pending_new",
                                              "partially_filled")]
            self.m_orders.set(str(len(open_orders)))
        else:
            self.m_broker.set("●  OFFLINE", sub=(snap.get("error") or "")[:42],
                              color=COLORS["red"])

        # Agent status
        if self._agent_running():
            pid = AGENT_PID_FILE.read_text().strip()
            self.agent_status.set(f"Running (PID {pid})", "ok")
        elif AGENT_STOP_FILE.exists():
            self.agent_status.set("Stop file present", "warn")
        else:
            self.agent_status.set("Stopped", "neutral")

        # API keys
        for k, lbl in self._api_labels.items():
            present = bool(os.getenv(k))
            lbl.configure(
                text=f"{'●' if present else '○'}  {k}",
                text_color=COLORS["green"] if present else COLORS["text_dim"],
            )

        # Log tail
        self._set_textbox(self.log_view, _tail(APP_LOG_FILE, 80))

        # Signals
        rows = []
        if SIGNALS_DIR.exists():
            paths = sorted(SIGNALS_DIR.glob("*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:14]
            for p in paths:
                try:
                    d = json.loads(p.read_text())
                    rows.append(
                        f"{p.stem:36s}  {d.get('asset','?'):5s}  "
                        f"{d.get('direction','?'):5s}  "
                        f"@ ${d.get('entry_price',0):>9,.2f}  "
                        f"conf {d.get('confidence',0):>4.0%}"
                    )
                except Exception:
                    continue
        self._set_textbox(self.signals_view,
                          "\n".join(rows) if rows else "(no signal records yet)")

    @staticmethod
    def _set_textbox(box: ctk.CTkTextbox, text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")


# ---------------------------------------------------------------------------
# Page: Positions / orders
# ---------------------------------------------------------------------------
class PositionsPage(ctk.CTkFrame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, fg_color=COLORS["bg"])
        self.app = app
        for i in range(3):
            self.grid_columnconfigure(i, weight=1)

        self.m_pnl = Metric(self, "Unrealized P&L", "$0.00")
        self.m_mv  = Metric(self, "Position Market Value", "$0.00")
        self.m_n   = Metric(self, "Positions Held", "0")
        self.m_pnl.grid(row=0, column=0, padx=(0, 8), pady=(0, 12), sticky="nsew")
        self.m_mv.grid(row=0, column=1, padx=8,   pady=(0, 12), sticky="nsew")
        self.m_n.grid(row=0, column=2, padx=(8, 0), pady=(0, 12), sticky="nsew")

        # Positions table
        pos_card = Card(self)
        pos_card.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(0, 12))
        pos_card.grid_rowconfigure(1, weight=1)
        pos_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(pos_card, text="OPEN POSITIONS",
                     text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")
        self.pos_tree = _make_treeview(
            pos_card,
            columns=("symbol", "side", "qty", "entry", "current",
                     "market_value", "pnl_$", "pnl_%"),
        )
        self.pos_tree.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        # Orders table
        ord_card = Card(self)
        ord_card.grid(row=2, column=0, columnspan=3, sticky="nsew")
        ord_card.grid_rowconfigure(1, weight=1)
        ord_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ord_card, text="RECENT ORDERS",
                     text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")
        self.ord_tree = _make_treeview(
            ord_card,
            columns=("submitted", "symbol", "side", "qty", "filled",
                     "type", "status", "id"),
        )
        self.ord_tree.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        self.grid_rowconfigure(1, weight=2)
        self.grid_rowconfigure(2, weight=2)

    def update_from(self, snap: Dict[str, Any]) -> None:
        if not snap.get("ok"):
            return

        positions = snap["positions"]
        equity = snap["equity"] or 1.0
        total_pnl = sum(p["unrealized_pl"] for p in positions)
        total_mv = sum(p["market_value"] for p in positions)

        self.m_pnl.set(
            f"${total_pnl:+,.2f}",
            sub=f"{(total_pnl / equity * 100):+.2f}% of account",
            color=COLORS["green"] if total_pnl >= 0 else COLORS["red"],
        )
        self.m_mv.set(f"${total_mv:,.2f}")
        self.m_n.set(str(len(positions)))

        # Repopulate positions tree
        self.pos_tree.delete(*self.pos_tree.get_children())
        for p in positions:
            tag = "gain" if p["unrealized_pl"] >= 0 else "loss"
            self.pos_tree.insert(
                "", "end", tags=(tag,),
                values=(
                    p["symbol"], p["side"], f"{p['qty']:g}",
                    f"${p['avg_entry_price']:,.2f}",
                    f"${p['current_price']:,.2f}",
                    f"${p['market_value']:,.2f}",
                    f"${p['unrealized_pl']:+,.2f}",
                    f"{p['unrealized_plpc'] * 100:+.2f}%",
                ),
            )

        # Repopulate orders tree
        self.ord_tree.delete(*self.ord_tree.get_children())
        for o in snap["orders"]:
            status_tag = "open" if o["status"] in (
                "new", "accepted", "pending_new", "partially_filled"
            ) else "closed"
            self.ord_tree.insert(
                "", "end", tags=(status_tag,),
                values=(
                    (o["submitted_at"] or "")[:19],
                    o["symbol"], o["side"], f"{o['qty']:g}",
                    f"{o['filled_qty']:g}", o["order_type"],
                    o["status"], o["id"][:10] + "…",
                ),
            )


# ---------------------------------------------------------------------------
# Page: Trade controls
# ---------------------------------------------------------------------------
class ControlsPage(ctk.CTkFrame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent, fg_color=COLORS["bg"])
        self.app = app
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)

        # ----- New order card ----------
        form = Card(self)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 12))
        form.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(form, text="NEW ORDER", text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, columnspan=2, padx=16, pady=(14, 6),
                            sticky="w")

        def add_field(row, label, default=""):
            ctk.CTkLabel(form, text=label, text_color=COLORS["text_muted"],
                         font=ctk.CTkFont(family=SANS_FONT[0], size=11),
                         anchor="w"
                         ).grid(row=row, column=0, padx=(16, 8), pady=4, sticky="w")
            var = ctk.StringVar(value=default)
            ent = ctk.CTkEntry(form, textvariable=var, fg_color=COLORS["bg"],
                                border_color=COLORS["border"], height=32)
            ent.grid(row=row, column=1, padx=(0, 16), pady=4, sticky="ew")
            return var

        self.v_symbol = add_field(1, "Symbol", "SPY")

        ctk.CTkLabel(form, text="Side", text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11),
                     anchor="w"
                     ).grid(row=2, column=0, padx=(16, 8), pady=4, sticky="w")
        self.v_side = ctk.StringVar(value="long")
        side_frame = ctk.CTkFrame(form, fg_color="transparent")
        side_frame.grid(row=2, column=1, padx=(0, 16), pady=4, sticky="ew")
        ctk.CTkRadioButton(side_frame, text="Long", variable=self.v_side,
                           value="long", fg_color=COLORS["green"]
                           ).pack(side="left", padx=(0, 14))
        ctk.CTkRadioButton(side_frame, text="Short", variable=self.v_side,
                           value="short", fg_color=COLORS["red"]
                           ).pack(side="left")

        self.v_notional = add_field(3, "Notional ($)", "1000")
        self.v_stop = add_field(4, "Stop %", "1.0")
        self.v_tp = add_field(5, "Take-profit %", "2.0")

        self.v_confirm = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(form, text="I have reviewed the parameters",
                        variable=self.v_confirm,
                        fg_color=COLORS["accent"]
                        ).grid(row=6, column=0, columnspan=2,
                               padx=16, pady=(10, 4), sticky="w")

        self.btn_submit = ctk.CTkButton(
            form, text="Place Bracket Order",
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hi"],
            font=ctk.CTkFont(family=SANS_FONT[0], size=13, weight="bold"),
            height=40, command=self._submit,
        )
        self.btn_submit.grid(row=7, column=0, columnspan=2,
                             padx=16, pady=(8, 14), sticky="ew")

        # ----- Positions / orders side ----------
        side = ctk.CTkFrame(self, fg_color="transparent")
        side.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        side.grid_rowconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1)
        side.grid_columnconfigure(0, weight=1)

        self.pos_card = Card(side)
        self.pos_card.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.pos_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.pos_card, text="CLOSE POSITIONS",
                     text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")
        self.pos_scroll = ctk.CTkScrollableFrame(self.pos_card,
                                                  fg_color="transparent")
        self.pos_scroll.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.pos_scroll.grid_columnconfigure(0, weight=1)
        self.pos_card.grid_rowconfigure(1, weight=1)

        self.ord_card = Card(side)
        self.ord_card.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.ord_card.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(self.ord_card, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="CANCEL ORDERS",
                     text_color=COLORS["text_muted"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=11, weight="bold")
                     ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(hdr, text="⚠  Cancel ALL",
                      fg_color=COLORS["red_dim"], hover_color=COLORS["red"],
                      width=130, command=self._cancel_all
                      ).grid(row=0, column=1, sticky="e")
        self.ord_scroll = ctk.CTkScrollableFrame(self.ord_card,
                                                  fg_color="transparent")
        self.ord_scroll.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.ord_scroll.grid_columnconfigure(0, weight=1)
        self.ord_card.grid_rowconfigure(1, weight=1)

        self.grid_rowconfigure(0, weight=1)

    # ---- update --------------------------------------------------------
    def update_from(self, snap: Dict[str, Any]) -> None:
        if not snap.get("ok"):
            return
        self._executor = snap["executor"]

        # Rebuild positions list
        for child in self.pos_scroll.winfo_children():
            child.destroy()
        if not snap["positions"]:
            ctk.CTkLabel(self.pos_scroll, text="No open positions.",
                         text_color=COLORS["text_dim"]
                         ).grid(row=0, column=0, padx=4, pady=8, sticky="w")
        for i, p in enumerate(snap["positions"]):
            row = ctk.CTkFrame(self.pos_scroll, fg_color=COLORS["panel_hi"]
                                if i % 2 else COLORS["panel"],
                                corner_radius=6)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(0, weight=1)
            txt = (f"  {p['symbol']:<6} {p['side']:<5}  qty {p['qty']:>5g}   "
                   f"P&L ${p['unrealized_pl']:+,.2f}")
            color = COLORS["green"] if p["unrealized_pl"] >= 0 else COLORS["red"]
            ctk.CTkLabel(row, text=txt, anchor="w",
                         text_color=color,
                         font=ctk.CTkFont(family=MONO_FONT[0], size=12)
                         ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
            ctk.CTkButton(row, text="Close", width=80,
                          fg_color=COLORS["red_dim"], hover_color=COLORS["red"],
                          command=lambda s=p["symbol"]: self._close(s)
                          ).grid(row=0, column=1, padx=8, pady=4)

        # Rebuild orders list (open ones only)
        for child in self.ord_scroll.winfo_children():
            child.destroy()
        open_orders = [o for o in snap["orders"]
                       if o["status"] in ("new", "accepted", "pending_new",
                                          "partially_filled")]
        if not open_orders:
            ctk.CTkLabel(self.ord_scroll, text="No open orders.",
                         text_color=COLORS["text_dim"]
                         ).grid(row=0, column=0, padx=4, pady=8, sticky="w")
        for i, o in enumerate(open_orders):
            row = ctk.CTkFrame(self.ord_scroll, fg_color=COLORS["panel_hi"]
                                if i % 2 else COLORS["panel"],
                                corner_radius=6)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(0, weight=1)
            txt = (f"  {o['symbol']:<6} {o['side']:<5} qty {o['qty']:>5g}  "
                   f"{o['order_type']:<8} ({o['status']})")
            ctk.CTkLabel(row, text=txt, anchor="w",
                         text_color=COLORS["text"],
                         font=ctk.CTkFont(family=MONO_FONT[0], size=12)
                         ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
            ctk.CTkButton(row, text="Cancel", width=80,
                          fg_color=COLORS["yellow"],
                          hover_color="#fbbf24", text_color=COLORS["bg"],
                          command=lambda oid=o["id"]: self._cancel(oid)
                          ).grid(row=0, column=1, padx=8, pady=4)

    # ---- actions ------------------------------------------------------
    def _submit(self) -> None:
        if not self.v_confirm.get():
            self.app.toast("Tick the 'reviewed' box first.")
            return
        symbol = self.v_symbol.get().strip().upper()
        if not symbol:
            self.app.toast("Symbol required.")
            return
        try:
            notional = float(self.v_notional.get())
            stop_pct = float(self.v_stop.get()) / 100.0
            tp_pct = float(self.v_tp.get()) / 100.0
        except ValueError:
            self.app.toast("Numeric fields must be numbers.")
            return

        # Pull latest trade.
        import requests
        try:
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
            self.app.toast(f"Quote lookup failed: {exc}")
            return
        if entry <= 0:
            self.app.toast("Got zero/missing price — symbol typo?")
            return

        direction = self.v_side.get()
        stop = entry * (1 - stop_pct) if direction == "long" else entry * (1 + stop_pct)
        tp = entry * (1 + tp_pct) if direction == "long" else entry * (1 - tp_pct)

        sig = {
            "asset": symbol,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "direction": direction,
            "entry_price": round(entry, 2),
            "stop_loss": round(stop, 2),
            "take_profit": round(tp, 2),
            "position_size_usd": notional,
            "expected_return_pct": tp_pct,
            "iv_change_pct": 0.0,
            "confidence": 1.0,
            "risk_flags": {"manual_desktop": True},
        }
        try:
            ex = self._executor or AlpacaExecutor(live_money=False)
            result = ex.submit_signal(sig)
            if result.submitted:
                self.app.toast(f"✓  Submitted — id {result.order_id[:10]}…")
                self.v_confirm.set(False)
                self.app.poller.trigger()
            else:
                self.app.toast(f"✗  Rejected: {result.reason}")
        except Exception as exc:  # noqa: BLE001
            self.app.toast(f"Submission failed: {exc}")

    def _close(self, symbol: str) -> None:
        if not self._executor:
            return
        ok = self._executor.close_position(symbol)
        self.app.toast(f"{'Close requested for' if ok else 'Close FAILED:'} {symbol}")
        self.app.poller.trigger()

    def _cancel(self, order_id: str) -> None:
        if not self._executor:
            return
        ok = self._executor.cancel_order(order_id)
        self.app.toast("Cancel requested." if ok else "Cancel failed.")
        self.app.poller.trigger()

    def _cancel_all(self) -> None:
        if not self._executor:
            return
        n = self._executor.cancel_all_orders()
        self.app.toast(f"Cancelled {n} order(s).")
        self.app.poller.trigger()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("trading_enhancer")
        self.geometry("1320x860")
        self.minsize(1100, 720)
        self.configure(fg_color=COLORS["bg"])

        # Layout: sidebar (220) | content
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content()
        self._build_statusbar()

        self.queue: Queue = Queue()
        self.poller = BrokerPoller(self.queue, interval=8.0)
        self.poller.start()
        self.after(150, self._drain_queue)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- layout pieces ------------------------------------------------
    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, fg_color=COLORS["panel"], width=220,
                                corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsw")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)

        # Brand
        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=18, pady=(20, 12))
        ctk.CTkLabel(brand, text="◆  trading_enhancer",
                     font=ctk.CTkFont(family=SANS_FONT[0], size=15, weight="bold"),
                     text_color=COLORS["text"]
                     ).pack(anchor="w")
        ctk.CTkLabel(brand, text="autonomous control",
                     font=ctk.CTkFont(family=SANS_FONT[0], size=10),
                     text_color=COLORS["text_dim"]
                     ).pack(anchor="w")

        # Nav buttons
        self.nav_buttons: Dict[str, ctk.CTkButton] = {}
        for i, (key, icon, label) in enumerate([
            ("health", "🩺", "Health"),
            ("positions", "💹", "Positions"),
            ("controls", "🛒", "Controls"),
        ]):
            btn = ctk.CTkButton(
                sidebar, text=f"   {icon}    {label}",
                anchor="w", height=42,
                fg_color="transparent", hover_color=COLORS["panel_hi"],
                text_color=COLORS["text_muted"],
                font=ctk.CTkFont(family=SANS_FONT[0], size=13),
                corner_radius=6,
                command=lambda k=key: self._navigate(k),
            )
            btn.grid(row=1 + i, column=0, padx=12, pady=2, sticky="ew")
            self.nav_buttons[key] = btn

        # Spacer + footer
        sidebar.grid_rowconfigure(99, weight=1)
        foot = ctk.CTkFrame(sidebar, fg_color="transparent")
        foot.grid(row=100, column=0, sticky="ew", padx=16, pady=14)
        ctk.CTkLabel(foot, text="v0.1 • paper mode",
                     text_color=COLORS["text_dim"],
                     font=ctk.CTkFont(family=SANS_FONT[0], size=10)
                     ).pack(anchor="w")

    def _build_content(self) -> None:
        self.content = ctk.CTkFrame(self, fg_color=COLORS["bg"])
        self.content.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.pages: Dict[str, ctk.CTkFrame] = {
            "health": HealthPage(self.content, self),
            "positions": PositionsPage(self.content, self),
            "controls": ControlsPage(self.content, self),
        }
        for p in self.pages.values():
            p.grid(row=0, column=0, sticky="nsew")
        self._navigate("health")

    def _build_statusbar(self) -> None:
        self.statusbar = ctk.CTkFrame(self, fg_color=COLORS["panel"],
                                       height=26, corner_radius=0)
        self.statusbar.grid(row=1, column=1, sticky="ew")
        self.statusbar.grid_columnconfigure(1, weight=1)
        self.toast_label = ctk.CTkLabel(
            self.statusbar, text="Ready.", anchor="w",
            text_color=COLORS["text_dim"],
            font=ctk.CTkFont(family=SANS_FONT[0], size=11),
        )
        self.toast_label.grid(row=0, column=0, padx=14, sticky="w")
        self.clock_label = ctk.CTkLabel(
            self.statusbar, text="",
            text_color=COLORS["text_dim"],
            font=ctk.CTkFont(family=MONO_FONT[0], size=11),
        )
        self.clock_label.grid(row=0, column=2, padx=14, sticky="e")
        self._tick_clock()

    # ---- navigation ---------------------------------------------------
    def _navigate(self, key: str) -> None:
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=COLORS["accent"],
                              text_color=COLORS["text"])
            else:
                btn.configure(fg_color="transparent",
                              text_color=COLORS["text_muted"])
        self.pages[key].tkraise()

    # ---- helpers ------------------------------------------------------
    def toast(self, message: str) -> None:
        self.toast_label.configure(text=message, text_color=COLORS["text"])
        self.after(5000,
                   lambda: self.toast_label.configure(
                       text="Ready.", text_color=COLORS["text_dim"]))

    def _tick_clock(self) -> None:
        self.clock_label.configure(
            text=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
        self.after(1000, self._tick_clock)

    def _drain_queue(self) -> None:
        snap = None
        while True:
            try:
                snap = self.queue.get_nowait()
            except Empty:
                break
        if snap is not None:
            for p in self.pages.values():
                try:
                    p.update_from(snap)
                except Exception:
                    traceback.print_exc()
        self.after(300, self._drain_queue)

    def _on_close(self) -> None:
        try:
            self.poller.stop()
        except Exception:
            pass
        self.destroy()


# ---------------------------------------------------------------------------
# Treeview helpers
# ---------------------------------------------------------------------------
def _make_treeview(parent, columns: tuple[str, ...]) -> ttk.Treeview:
    """Style a ttk.Treeview to match the dark theme. ttk lives in
    stdlib so we don't add another dependency for the tables."""
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Treeview",
        background=COLORS["bg"], foreground=COLORS["text"],
        fieldbackground=COLORS["bg"], borderwidth=0, rowheight=26,
        font=(MONO_FONT[0], 11),
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["panel_hi"], foreground=COLORS["text_muted"],
        font=(SANS_FONT[0], 10, "bold"), relief="flat",
    )
    style.map("Treeview",
              background=[("selected", COLORS["accent"])],
              foreground=[("selected", COLORS["text"])])

    tree = ttk.Treeview(parent, columns=columns, show="headings",
                        selectmode="browse")
    for c in columns:
        tree.heading(c, text=c.replace("_", " ").upper())
        tree.column(c, anchor="w", width=110, stretch=True)
    tree.tag_configure("gain", foreground=COLORS["green"])
    tree.tag_configure("loss", foreground=COLORS["red"])
    tree.tag_configure("open", foreground=COLORS["yellow"])
    tree.tag_configure("closed", foreground=COLORS["text_dim"])
    return tree


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return f"(no {path.name} yet)"
    try:
        lines = path.read_text(errors="replace").splitlines()[-n:]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"(read failed: {exc})"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
