"""Status page — system health, agent control, log tail, recent signals."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS
from brzrkr_app.widgets import (
    BloodMetric, CodexBox, GhostButton, GothicCard, PageTitle,
    PulseBar, RuneButton, SectionHeader, StatusBeacon,
)

ROOT = Path(__file__).resolve().parent.parent.parent
AGENT_PID = ROOT / "agent.pid"
AGENT_STOP = ROOT / "AGENT_STOP"
AGENT_LOG = ROOT / "agent.out"
APP_LOG = ROOT / "trading_enhancer.log"
AUTO_PID = ROOT / ".auto_trader.pid"
AUTO_STOP = ROOT / "AUTO_TRADER_STOP"
AUTO_LOG = ROOT / "auto_trader.out"
SIGNALS_DIR = ROOT / "data" / "signals"


class StatusPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app

        for i in range(4):
            self.grid_columnconfigure(i, weight=1)

        # Title
        PageTitle(self, f"{G.CROSS} Status",
                   subtitle="forge · agent · signal").grid(
            row=0, column=0, columnspan=4, sticky="ew", pady=(0, 12))

        # Top metrics
        self.m_broker = BloodMetric(self, "Broker", "—", "Connecting…")
        self.m_equity = BloodMetric(self, "War Chest", "$ —")
        self.m_pos = BloodMetric(self, "Open Positions", "0")
        self.m_ord = BloodMetric(self, "Open Orders", "0")
        self.m_broker.grid(row=1, column=0, padx=(0, 6), sticky="nsew")
        self.m_equity.grid(row=1, column=1, padx=6, sticky="nsew")
        self.m_pos.grid(row=1, column=2, padx=6, sticky="nsew")
        self.m_ord.grid(row=1, column=3, padx=(6, 0), sticky="nsew")

        # Agent control card
        agent_card = GothicCard(self)
        agent_card.grid(row=2, column=0, columnspan=4, sticky="nsew",
                         pady=(12, 12))
        agent_card.grid_columnconfigure(0, weight=1)
        SectionHeader(agent_card, "Agent Sigil", glyph=G.RUNE_T).grid(
            row=0, column=0, sticky="ew")

        body = ctk.CTkFrame(agent_card, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
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
            text_color=C.PARCHMENT,
            border_color=C.BORDER,
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

        # ---- Auto-Trader (market-aware daemon) ----------------------
        auto_card = GothicCard(self)
        auto_card.grid(row=2, column=0, columnspan=4, sticky="ew",
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
                         text_color=C.PARCHMENT, border_color=C.BORDER
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
                    command=self._start_auto_trader
                    ).grid(row=0, column=1, padx=(4, 4))
        GhostButton(ab, "Stop", glyph=G.SHIELD,
                     command=self._stop_auto_trader
                     ).grid(row=0, column=2, padx=(4, 0))

        # Recent signals
        sig = GothicCard(self)
        sig.grid(row=3, column=0, columnspan=2, sticky="nsew",
                  padx=(0, 6), pady=(0, 0))
        sig.grid_columnconfigure(0, weight=1)
        sig.grid_rowconfigure(1, weight=1)
        SectionHeader(sig, "Codex of Signals", glyph=G.RUNE_F).grid(
            row=0, column=0, sticky="ew")
        self.signals_box = CodexBox(sig, height=240)
        self.signals_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

        # Log tail
        log = GothicCard(self)
        log.grid(row=3, column=2, columnspan=2, sticky="nsew",
                  padx=(6, 0), pady=(0, 0))
        log.grid_columnconfigure(0, weight=1)
        log.grid_rowconfigure(1, weight=1)
        SectionHeader(log, "Whispered Log", glyph=G.RUNE_O).grid(
            row=0, column=0, sticky="ew")
        self.log_box = CodexBox(log, height=240)
        self.log_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

        self.grid_rowconfigure(3, weight=1)

    # ------------------------------------------------------------------
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

    # ---- Auto-Trader controls --------------------------------------
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

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
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

        # Auto-Trader beacon
        running, auto_pid = self._auto_running()
        if running:
            self.auto_beacon.set(
                f"Auto-Trader running  ·  PID {auto_pid}", "ok")
        elif AUTO_STOP.exists():
            self.auto_beacon.set("Auto-Trader stopping", "warn")
        else:
            self.auto_beacon.set("Auto-Trader idle", "neutral",
                                  glyph=G.DOT_DIM)
        # Show next-market-open from broker if available
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

        # Signals box
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

        # Log tail
        text = _tail(APP_LOG, 60)
        self.log_box.set_text(text)


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return f"{G.DOT_DIM}  no {path.name} yet"
    try:
        lines = path.read_text(errors="replace").splitlines()[-n:]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"(read failed: {exc})"
