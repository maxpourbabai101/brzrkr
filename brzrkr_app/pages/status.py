"""Status page — system health, agent control, log tail, recent signals,
and the full trade console (order form + position sealer + order banisher).

The Console tab has been merged here so everything lives in one place.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, pnl_color
from brzrkr_app.widgets import (
    BloodMetric, CodexBox, GhostButton, GothicCard, PageTitle,
    PulseBar, RuneButton, SectionHeader, StatusBeacon,
)

ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_PID  = ROOT / "agent.pid"
AGENT_STOP = ROOT / "AGENT_STOP"
AGENT_LOG  = ROOT / "agent.out"
APP_LOG    = ROOT / "trading_enhancer.log"
AUTO_PID   = ROOT / ".auto_trader.pid"
AUTO_STOP  = ROOT / "AUTO_TRADER_STOP"
AUTO_LOG   = ROOT / "auto_trader.out"
SIGNALS_DIR = ROOT / "data" / "signals"


class StatusPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._executor = None   # wired in via update_from → console ops

        # Outer frame fills its grid cell; inner scrollable holds all content
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._inner = ctk.CTkScrollableFrame(
            self, fg_color=C.NIGHT,
            scrollbar_button_color=C.BLOOD_DIM,
            scrollbar_button_hover_color=C.BLOOD,
        )
        self._inner.grid(row=0, column=0, sticky="nsew")
        for i in range(4):
            self._inner.grid_columnconfigure(i, weight=1)

        self._build_status_section()
        self._build_console_section()

    # ================================================================
    # STATUS SECTION
    # ================================================================
    def _build_status_section(self) -> None:
        inner = self._inner

        # ── Row 0 : title ──────────────────────────────────────────
        PageTitle(inner, f"{G.CROSS} Status",
                  subtitle="forge · agent · signal · console").grid(
            row=0, column=0, columnspan=4, sticky="ew", pady=(0, 12))

        # ── Row 1 : broker metrics ─────────────────────────────────
        self.m_broker = BloodMetric(inner, "Broker",         "—", "Connecting…")
        self.m_equity = BloodMetric(inner, "War Chest",      "$ —")
        self.m_pos    = BloodMetric(inner, "Open Positions",  "0")
        self.m_ord    = BloodMetric(inner, "Open Orders",     "0")
        self.m_broker.grid(row=1, column=0, padx=(0, 6),  sticky="nsew")
        self.m_equity.grid(row=1, column=1, padx=6,        sticky="nsew")
        self.m_pos.grid   (row=1, column=2, padx=6,        sticky="nsew")
        self.m_ord.grid   (row=1, column=3, padx=(6, 0),   sticky="nsew")

        # ── Row 2 : agent control card ─────────────────────────────
        agent_card = GothicCard(inner)
        agent_card.grid(row=2, column=0, columnspan=4, sticky="nsew",
                        pady=(12, 6))
        agent_card.grid_columnconfigure(0, weight=1)
        SectionHeader(agent_card, "Agent Sigil", glyph=G.RUNE_T).grid(
            row=0, column=0, sticky="ew")

        body = ctk.CTkFrame(agent_card, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        body.grid_columnconfigure(0, weight=1)

        self.agent_beacon = StatusBeacon(body, "Checking…", "neutral")
        self.agent_beacon.grid(row=0, column=0, sticky="w")
        self.pulse = PulseBar(body, width=200, height=14)
        self.pulse.grid(row=0, column=1, sticky="e", padx=8)

        controls = ctk.CTkFrame(agent_card, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        for i in range(6):
            controls.grid_columnconfigure(i, weight=1, uniform="ctrl")

        self.var_dry = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            controls, text="Dry run (no orders)", variable=self.var_dry,
            fg_color=C.BLOOD, hover_color=C.BLOOD_HI,
            text_color=C.PARCHMENT, border_color=C.BORDER,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ctk.CTkLabel(
            controls, text="● PAPER — virtual funds only",
            text_color=C.LIFE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=9),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        self.var_tick = ctk.StringVar(value="60")
        ctk.CTkEntry(controls, textvariable=self.var_tick, width=70,
                     fg_color=C.OBSIDIAN, border_color=C.BORDER,
                     text_color=C.BONE).grid(
            row=0, column=2, sticky="w", padx=4)

        self.var_source = ctk.StringVar(value="api")
        ctk.CTkOptionMenu(
            controls, values=["api", "scraper", "both"],
            variable=self.var_source, width=110,
            fg_color=C.PANEL_HI, button_color=C.BLOOD_DIM,
            button_hover_color=C.BLOOD,
            text_color=C.BONE, dropdown_fg_color=C.PANEL,
        ).grid(row=0, column=3, sticky="w")

        RuneButton(controls, "Start", glyph=G.EXEC,
                   command=self.start_agent).grid(
            row=0, column=4, sticky="ew", padx=(8, 4))
        GhostButton(controls, "Stop", glyph=G.SHIELD,
                    command=self.stop_agent).grid(
            row=0, column=5, sticky="ew", padx=(4, 0))

        # ── Row 3 : auto-trader card ───────────────────────────────
        auto_card = GothicCard(inner)
        auto_card.grid(row=3, column=0, columnspan=4, sticky="ew",
                       pady=(0, 12))
        auto_card.grid_columnconfigure(0, weight=1)
        SectionHeader(auto_card, "Auto-Trader  (market-open daemon)",
                      glyph=G.SUN).grid(row=0, column=0, sticky="ew")
        ac = ctk.CTkFrame(auto_card, fg_color="transparent")
        ac.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        ac.grid_columnconfigure(0, weight=1)

        self.auto_beacon = StatusBeacon(ac, "checking…", "neutral")
        self.auto_beacon.grid(row=0, column=0, sticky="w")
        self.auto_next = ctk.CTkLabel(
            ac, text="", text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10), anchor="w",
        )
        self.auto_next.grid(row=1, column=0, sticky="w", pady=(2, 6))

        self.auto_dry = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(ac, text="Dry run (no orders)", variable=self.auto_dry,
                        fg_color=C.BLOOD, hover_color=C.BLOOD_HI,
                        text_color=C.PARCHMENT, border_color=C.BORDER,
                        ).grid(row=2, column=0, sticky="w")
        ctk.CTkLabel(
            ac, text="● PAPER — virtual funds only · safe to execute",
            text_color=C.LIFE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=9),
        ).grid(row=3, column=0, sticky="w", pady=(2, 0))

        ab = ctk.CTkFrame(ac, fg_color="transparent")
        ab.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        ab.grid_columnconfigure(0, weight=1)
        RuneButton(ab, "Start Auto-Trader", glyph=G.SUN,
                   command=self._start_auto_trader,
                   ).grid(row=0, column=1, padx=(4, 4))
        GhostButton(ab, "Stop", glyph=G.SHIELD,
                    command=self._stop_auto_trader,
                    ).grid(row=0, column=2, padx=(4, 0))

        # ── Row 4 : signals + log tail ─────────────────────────────
        sig = GothicCard(inner)
        sig.grid(row=4, column=0, columnspan=2, sticky="nsew",
                 padx=(0, 6), pady=(0, 0))
        sig.grid_columnconfigure(0, weight=1)
        sig.grid_rowconfigure(1, weight=1)
        SectionHeader(sig, "Codex of Signals", glyph=G.RUNE_F).grid(
            row=0, column=0, sticky="ew")
        self.signals_box = CodexBox(sig, height=200)
        self.signals_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

        log = GothicCard(inner)
        log.grid(row=4, column=2, columnspan=2, sticky="nsew",
                 padx=(6, 0), pady=(0, 0))
        log.grid_columnconfigure(0, weight=1)
        log.grid_rowconfigure(1, weight=1)
        SectionHeader(log, "Whispered Log", glyph=G.RUNE_O).grid(
            row=0, column=0, sticky="ew")
        self.log_box = CodexBox(log, height=200)
        self.log_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

    # ================================================================
    # CONSOLE SECTION  (merged from ConsolePage)
    # ================================================================
    def _build_console_section(self) -> None:
        inner = self._inner

        # ── Row 5 : divider ────────────────────────────────────────
        ctk.CTkFrame(inner, height=2, fg_color=C.BORDER).grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=(18, 0))

        # ── Row 6 : console header ─────────────────────────────────
        hdr_row = ctk.CTkFrame(inner, fg_color="transparent")
        hdr_row.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(6, 8))
        hdr_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hdr_row,
            text=f"  {G.EXEC}  TRADE CONSOLE   ·   raise · seal · banish",
            text_color=C.BLOOD_HI,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        # ── Row 7 : form (left) + positions/orders (right) ─────────
        # Left — Forge New Order
        form = GothicCard(inner)
        form.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=(0, 6))
        form.grid_columnconfigure(1, weight=1)
        SectionHeader(form, "Forge New Order", glyph=G.RUNE_F).grid(
            row=0, column=0, columnspan=2, sticky="ew")

        def field(row, label, default=""):
            ctk.CTkLabel(form, text=label.upper(),
                         text_color=C.PARCHMENT,
                         font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
                         anchor="w").grid(row=row, column=0, padx=(16, 8), pady=4, sticky="w")
            v = ctk.StringVar(value=default)
            e = ctk.CTkEntry(form, textvariable=v, fg_color=C.OBSIDIAN,
                             border_color=C.BORDER, text_color=C.BONE, height=30)
            e.grid(row=row, column=1, padx=(0, 16), pady=4, sticky="ew")
            return v

        self.v_symbol   = field(1, "Symbol",         "SPY")

        ctk.CTkLabel(form, text="SIDE", text_color=C.PARCHMENT,
                     font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
                     anchor="w").grid(row=2, column=0, padx=(16, 8), pady=4, sticky="w")
        self.v_side = ctk.StringVar(value="long")
        side_f = ctk.CTkFrame(form, fg_color="transparent")
        side_f.grid(row=2, column=1, padx=(0, 16), pady=4, sticky="ew")
        ctk.CTkRadioButton(side_f, text="Long",  variable=self.v_side, value="long",
                           fg_color=C.LIFE,  text_color=C.BONE, border_color=C.BORDER,
                           ).pack(side="left", padx=(0, 14))
        ctk.CTkRadioButton(side_f, text="Short", variable=self.v_side, value="short",
                           fg_color=C.DEATH, text_color=C.BONE, border_color=C.BORDER,
                           ).pack(side="left")

        self.v_notional = field(3, "Notional ($)",  "1000")
        self.v_stop     = field(4, "Stop %",        "1.0")
        self.v_tp       = field(5, "Take Profit %", "2.0")

        ctk.CTkLabel(form, text="SESSION",
                     text_color=C.PARCHMENT,
                     font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
                     anchor="w").grid(row=6, column=0, padx=(16, 8), pady=4, sticky="w")
        self.v_session = ctk.StringVar(value="regular")
        sess_f = ctk.CTkFrame(form, fg_color="transparent")
        sess_f.grid(row=6, column=1, padx=(0, 16), pady=4, sticky="ew")
        for sess_val, sess_lbl, sess_col in [
            ("regular",     "Regular",    C.PARCHMENT),
            ("pre_market",  "Pre-Market", C.OMEN),
            ("after_hours", "After-Hours",C.OMEN),
        ]:
            ctk.CTkRadioButton(
                sess_f, text=sess_lbl, variable=self.v_session, value=sess_val,
                fg_color=C.BLOOD, hover_color=C.BLOOD_HI,
                border_color=C.BORDER, text_color=sess_col,
                font=ctk.CTkFont(family=FONT_SANS[0], size=10),
                command=self._on_session_change,
            ).pack(side="left", padx=(0, 10))

        self._ext_note = ctk.CTkLabel(
            form,
            text="  ⚠  Extended hours: limit order only, no bracket",
            text_color=C.OMEN,
            font=ctk.CTkFont(family=FONT_SANS[0], size=9),
            anchor="w",
        )

        self.v_confirm = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(form, text="I bear the consequence",
                        variable=self.v_confirm,
                        fg_color=C.BLOOD, hover_color=C.BLOOD_HI,
                        border_color=C.BORDER, text_color=C.PARCHMENT,
                        ).grid(row=8, column=0, columnspan=2,
                               padx=16, pady=(10, 4), sticky="w")

        RuneButton(form, "Strike", glyph=G.EXEC,
                   command=self._submit,
                   ).grid(row=9, column=0, columnspan=2,
                          padx=16, pady=(8, 14), sticky="ew")

        # Right — Seal Positions + Banish Orders stacked
        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.grid(row=7, column=2, columnspan=2, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        pos = GothicCard(right)
        pos.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        pos.grid_columnconfigure(0, weight=1)
        pos.grid_rowconfigure(1, weight=1)
        SectionHeader(pos, "Seal Positions", glyph=G.SHIELD).grid(
            row=0, column=0, sticky="ew")
        self.pos_scroll = ctk.CTkScrollableFrame(pos, fg_color="transparent", height=160)
        self.pos_scroll.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.pos_scroll.grid_columnconfigure(0, weight=1)

        ord_card = GothicCard(right)
        ord_card.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        ord_card.grid_columnconfigure(0, weight=1)
        ord_card.grid_rowconfigure(1, weight=1)
        hf = ctk.CTkFrame(ord_card, fg_color="transparent")
        hf.grid(row=0, column=0, sticky="ew")
        hf.grid_columnconfigure(0, weight=1)
        SectionHeader(hf, "Banish Orders", glyph=G.DAGGER).grid(
            row=0, column=0, sticky="ew")
        GhostButton(hf, "Banish ALL", glyph=G.SKULL,
                    command=self._cancel_all,
                    ).grid(row=0, column=1, padx=(0, 16), pady=(12, 4))
        self.ord_scroll = ctk.CTkScrollableFrame(ord_card, fg_color="transparent", height=160)
        self.ord_scroll.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.ord_scroll.grid_columnconfigure(0, weight=1)

    # ================================================================
    # Agent / Auto-Trader control
    # ================================================================
    def start_agent(self) -> None:
        if self._agent_running():
            self.app.toast("Agent already breathes.")
            return
        if AGENT_STOP.exists():
            try: AGENT_STOP.unlink()
            except Exception: pass
        tick = max(10, int(self.var_tick.get() or "60"))
        cmd = [sys.executable, str(ROOT / "agent.py"),
               "--dry-run" if self.var_dry.get() else "--execute",
               "--tick-seconds", str(tick),
               "--source", self.var_source.get()]
        log = open(AGENT_LOG, "ab")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(ROOT))
        AGENT_PID.write_text(str(proc.pid))
        self.app.toast(f"Agent risen (PID {proc.pid})")

    def stop_agent(self) -> None:
        if not self._agent_running():
            self.app.toast("Agent is silent.")
            return
        AGENT_STOP.touch()
        self.app.toast("Stop sigil scrawled. Falling within ~2s.")

    @staticmethod
    def _agent_running() -> bool:
        if not AGENT_PID.exists():
            return False
        try:
            pid = int(AGENT_PID.read_text().strip())
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    @staticmethod
    def _auto_running() -> tuple[bool, int]:
        if not AUTO_PID.exists():
            return (False, 0)
        try:
            pid = int(AUTO_PID.read_text().strip())
            os.kill(pid, 0)
            return (True, pid)
        except Exception:
            return (False, 0)

    def _start_auto_trader(self) -> None:
        running, _ = self._auto_running()
        if running:
            self.app.toast("Auto-Trader already running.")
            return
        if AUTO_STOP.exists():
            try: AUTO_STOP.unlink()
            except Exception: pass
        cmd = [sys.executable, str(ROOT / "auto_trader.py"),
               "--dry-run" if self.auto_dry.get() else "--execute",
               "--tick-seconds", "300"]
        log = open(AUTO_LOG, "ab")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(ROOT))
        AUTO_PID.write_text(str(proc.pid))
        self.app.toast(f"{G.SUN}  Auto-Trader online (PID {proc.pid}).")

    def _stop_auto_trader(self) -> None:
        running, _ = self._auto_running()
        if not running:
            self.app.toast("Auto-Trader isn't running.")
            return
        AUTO_STOP.touch()
        self.app.toast("Auto-Trader stop sigil written.")

    # ================================================================
    # Console — order form + position/order actions
    # ================================================================
    def _submit(self) -> None:
        if not self.v_confirm.get():
            self.app.toast("Bear the consequence first.")
            return
        symbol = self.v_symbol.get().strip().upper()
        if not symbol:
            self.app.toast("A name is required.")
            return
        try:
            notional = float(self.v_notional.get())
            stop_pct = float(self.v_stop.get()) / 100.0
            tp_pct   = float(self.v_tp.get()) / 100.0
        except ValueError:
            self.app.toast("Numeric fields must hold numbers.")
            return

        import requests
        try:
            r = requests.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                headers={
                    "APCA-API-KEY-ID":     os.getenv("ALPACA_API_KEY"),
                    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
                }, timeout=10)
            r.raise_for_status()
            entry = float((r.json().get("trade") or {}).get("p", 0))
        except Exception as exc:
            self.app.toast(f"Quote unreachable: {exc}")
            return
        if entry <= 0:
            self.app.toast("No price; symbol misspoken?")
            return

        direction = self.v_side.get()
        stop = entry * (1 - stop_pct) if direction == "long" else entry * (1 + stop_pct)
        tp   = entry * (1 + tp_pct)   if direction == "long" else entry * (1 - tp_pct)
        sig  = {
            "asset":              symbol,
            "timestamp":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "direction":          direction,
            "entry_price":        round(entry, 2),
            "stop_loss":          round(stop,  2),
            "take_profit":        round(tp,    2),
            "position_size_usd":  notional,
            "expected_return_pct": tp_pct,
            "iv_change_pct":      0.0,
            "confidence":         1.0,
            "risk_flags":         {"manual_brzrkr": True},
        }
        try:
            from src.execution.broker import AlpacaExecutor
            ex = self._executor or AlpacaExecutor(live_money=False)
            session    = self.v_session.get()
            ext_hours  = session in ("pre_market", "after_hours")
            result     = ex.submit_signal(sig, extended_hours=ext_hours, session=session)
            if result.submitted:
                sess_tag = f" [{session}]" if ext_hours else ""
                self.app.toast(f"{G.EXEC}  Struck{sess_tag} — id {result.order_id[:10]}…")
                self.v_confirm.set(False)
                self.app.poller.trigger()
            else:
                self.app.toast(f"{G.SKULL}  Rejected: {result.reason}")
        except Exception as exc:
            self.app.toast(f"Strike failed: {exc}")

    def _on_session_change(self) -> None:
        if self.v_session.get() in ("pre_market", "after_hours"):
            self._ext_note.grid(row=7, column=0, columnspan=2,
                                padx=16, pady=(0, 4), sticky="w")
        else:
            self._ext_note.grid_remove()

    def _close(self, symbol: str) -> None:
        if not self._executor:
            return
        ok = self._executor.close_position(symbol)
        self.app.toast(f"{'Sealed' if ok else 'SEAL FAILED:'} {symbol}")
        self.app.poller.trigger()

    def _cancel(self, order_id: str) -> None:
        if not self._executor:
            return
        ok = self._executor.cancel_order(order_id)
        self.app.toast("Banished." if ok else "Banish failed.")
        self.app.poller.trigger()

    def _cancel_all(self) -> None:
        if not self._executor:
            return
        n = self._executor.cancel_all_orders()
        self.app.toast(f"Banished {n} order(s).")
        self.app.poller.trigger()

    # ================================================================
    # Live update  (called by drain_queue every ~300 ms)
    # ================================================================
    def update_from(self, snap: dict) -> None:
        self._update_status(snap)
        self._update_console(snap)

    def _update_status(self, snap: dict) -> None:
        if snap.get("ok"):
            endpoint = "PAPER" if snap["paper"] else f"{G.SKULL} LIVE"
            self.m_broker.set(endpoint, sub="Connected", color=C.LIFE)
            self.m_equity.set(f"${snap['equity']:,.0f}")
            self.m_pos.set(str(len(snap["positions"])))
            open_orders = [o for o in snap["orders"]
                           if o["status"] in ("new", "accepted",
                                              "pending_new", "partially_filled")]
            self.m_ord.set(str(len(open_orders)))
            self.pulse.pulse(0.6 + 0.4 * (len(snap["positions"]) / 5.0))
        else:
            self.m_broker.set("SEVERED", sub=(snap.get("error") or "")[:42],
                              color=C.FORGE)
            self.pulse.pulse(0.1)

        if self._agent_running():
            pid = AGENT_PID.read_text().strip()
            self.agent_beacon.set(f"Agent rising  ·  PID {pid}", "alive")
        elif AGENT_STOP.exists():
            self.agent_beacon.set("Stop sigil present", "warn")
        else:
            self.agent_beacon.set("Agent dormant", "neutral", glyph=G.DOT_DIM)

        running, auto_pid = self._auto_running()
        if running:
            self.auto_beacon.set(f"Auto-Trader running  ·  PID {auto_pid}", "ok")
        elif AUTO_STOP.exists():
            self.auto_beacon.set("Auto-Trader stopping", "warn")
        else:
            self.auto_beacon.set("Auto-Trader idle", "neutral", glyph=G.DOT_DIM)

        try:
            if snap.get("executor") is not None:
                clk = snap["executor"]._client.get_clock()
                if getattr(clk, "is_open", False):
                    nxt = getattr(clk, "next_close", None)
                    self.auto_next.configure(
                        text=f"  {G.RIGHT}  market OPEN — closes at {nxt}")
                else:
                    nxt = getattr(clk, "next_open", None)
                    self.auto_next.configure(
                        text=f"  {G.RIGHT}  market closed — opens at {nxt}")
        except Exception:
            pass

        rows = []
        if SIGNALS_DIR.exists():
            paths = sorted(SIGNALS_DIR.glob("*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:14]
            for p in paths:
                try:
                    d = json.loads(p.read_text())
                    direction = d.get("direction", "?")
                    arrow = G.UP if direction == "long" else G.DOWN
                    rows.append(
                        f"{arrow}  {d.get('asset','?'):5s}  "
                        f"@ ${d.get('entry_price',0):>9,.2f}   "
                        f"stop ${d.get('stop_loss',0):>8,.2f}   "
                        f"tp ${d.get('take_profit',0):>8,.2f}   "
                        f"conf {d.get('confidence',0):>4.0%}"
                    )
                except Exception:
                    continue
        self.signals_box.set_text(
            "\n".join(rows) if rows
            else f"{G.DOT_DIM}  no signals recorded yet")

        self.log_box.set_text(_tail(APP_LOG, 60))

    def _update_console(self, snap: dict) -> None:
        if not snap.get("ok"):
            return
        self._executor = snap.get("executor")

        # Seal Positions panel
        for c in self.pos_scroll.winfo_children():
            c.destroy()
        if not snap["positions"]:
            ctk.CTkLabel(self.pos_scroll, text=f"{G.DOT_DIM}  no positions held",
                         text_color=C.ASH,
                         ).grid(row=0, column=0, padx=4, pady=8, sticky="w")
        for i, p in enumerate(snap["positions"]):
            row = ctk.CTkFrame(self.pos_scroll,
                               fg_color=C.PANEL_HI if i % 2 else C.PANEL,
                               corner_radius=2)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(0, weight=1)
            txt = (f"  {G.RIGHT}  {p['symbol']:<5} {p['side']:<5}  "
                   f"qty {p['qty']:>4g}    "
                   f"P&L ${p['unrealized_pl']:+,.2f}")
            ctk.CTkLabel(row, text=txt, anchor="w",
                         text_color=pnl_color(p["unrealized_pl"]),
                         font=ctk.CTkFont(family=FONT_MONO[0], size=12),
                         ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
            GhostButton(row, "Seal", width=70,
                        command=lambda s=p["symbol"]: self._close(s),
                        ).grid(row=0, column=1, padx=8, pady=4)

        # Banish Orders panel
        for c in self.ord_scroll.winfo_children():
            c.destroy()
        open_orders = [o for o in snap["orders"]
                       if o["status"] in ("new", "accepted",
                                          "pending_new", "partially_filled")]
        if not open_orders:
            ctk.CTkLabel(self.ord_scroll, text=f"{G.DOT_DIM}  no orders pending",
                         text_color=C.ASH,
                         ).grid(row=0, column=0, padx=4, pady=8, sticky="w")
        for i, o in enumerate(open_orders):
            row = ctk.CTkFrame(self.ord_scroll,
                               fg_color=C.PANEL_HI if i % 2 else C.PANEL,
                               corner_radius=2)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(0, weight=1)
            txt = (f"  {G.RIGHT}  {o['symbol']:<5} {o['side']:<4}  "
                   f"qty {o['qty']:>4g}  {o['order_type']:<8}  ({o['status']})")
            ctk.CTkLabel(row, text=txt, anchor="w",
                         text_color=C.BONE,
                         font=ctk.CTkFont(family=FONT_MONO[0], size=12),
                         ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
            GhostButton(row, "Banish", width=80,
                        command=lambda oid=o["id"]: self._cancel(oid),
                        ).grid(row=0, column=1, padx=8, pady=4)


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return f"{G.DOT_DIM}  no {path.name} yet"
    try:
        lines = path.read_text(errors="replace").splitlines()[-n:]
        return "\n".join(lines)
    except Exception as exc:
        return f"(read failed: {exc})"
