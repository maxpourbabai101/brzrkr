"""Main window — sidebar navigation, status bar, page stack.

Six pages: Status / Trades / Console / Market / Postmortem / Admin.
A background BrokerPoller pushes broker snapshots into a Queue; the
Tk event loop drains the queue and updates every visible page.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Dict

import customtkinter as ctk

from brzrkr_app.poller import BrokerPoller
from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS
from brzrkr_app.pages.status import StatusPage
from brzrkr_app.pages.trades import TradesPage
from brzrkr_app.pages.console import ConsolePage
from brzrkr_app.pages.market import MarketPage
from brzrkr_app.pages.research import ResearchPage
from brzrkr_app.pages.admin import AdminPage
from brzrkr_app.pages.postmortem import PostmortemPage
from brzrkr_app.pages.backtests import BacktestsPage
from brzrkr_app.pages.strategy import StrategyPage
from brzrkr_app.pages.system import SystemPage

ROOT = Path(__file__).resolve().parent.parent


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("dark")

        self.title("BRZRKR")
        self.geometry("1440x900")
        self.minsize(1200, 760)
        self.configure(fg_color=C.NIGHT)

        # Icon
        try:
            from brzrkr_app.icon import set_window_icon
            set_window_icon(self)
        except Exception:
            pass

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_content()
        self._build_statusbar()

        # Background poller.
        self.queue: Queue = Queue()
        self.poller = BrokerPoller(self.queue, interval=8.0)
        self.poller.start()
        self.after(150, self._drain_queue)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self, fg_color=C.VOID, width=240, corner_radius=0)
        sb.grid(row=0, column=0, rowspan=2, sticky="nsw")
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)

        # Brand
        brand = ctk.CTkFrame(sb, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=18, pady=(24, 8))

        ctk.CTkLabel(
            brand, text=f"{G.CROSS}  BRZRKR  {G.CROSS}",
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=20, weight="bold"),
            text_color=C.BLOOD_HI, anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand, text="The forge of trade",
            font=ctk.CTkFont(family=FONT_SANS[0], size=9, weight="bold"),
            text_color=C.ASH, anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        # Hairline
        ctk.CTkFrame(sb, height=1, fg_color=C.BORDER).grid(
            row=1, column=0, sticky="ew", padx=16, pady=(10, 14))

        # Nav buttons
        self.nav_buttons: Dict[str, ctk.CTkButton] = {}
        nav_items = [
            ("status",     G.RUNE_T,  "STATUS"),
            ("trades",     G.RUNE_F,  "TRADES"),
            ("console",    G.EXEC,    "CONSOLE"),
            ("market",     G.RUNE_R,  "MARKET"),
            ("research",   G.RUNE_O,  "RESEARCH"),
            ("backtests",  G.CROSS,   "BACKTESTS"),
            ("strategy",   G.ORNATE,  "STRATEGY"),
            ("postmortem", G.SKULL,   "POSTMORTEM"),
            ("system",     G.GEAR,    "SYSTEM"),
            ("admin",      G.DAGGER,  "ADMIN"),
        ]
        for i, (key, glyph, label) in enumerate(nav_items):
            btn = ctk.CTkButton(
                sb, text=f"    {glyph}     {label}",
                anchor="w", height=42,
                fg_color="transparent",
                hover_color=C.PANEL_HI,
                text_color=C.PARCHMENT,
                font=ctk.CTkFont(family=FONT_SANS[0], size=12, weight="bold"),
                corner_radius=2,
                command=lambda k=key: self._navigate(k),
            )
            btn.grid(row=2 + i, column=0, padx=10, pady=2, sticky="ew")
            self.nav_buttons[key] = btn

        # Footer
        sb.grid_rowconfigure(99, weight=1)
        foot = ctk.CTkFrame(sb, fg_color="transparent")
        foot.grid(row=100, column=0, sticky="ew", padx=16, pady=14)
        ctk.CTkLabel(
            foot, text=f"{G.DAGGER}  v0.1  ·  paper mode",
            text_color=C.GHOST,
            font=ctk.CTkFont(family=FONT_SANS[0], size=10),
        ).pack(anchor="w")

    def _build_content(self) -> None:
        self.content = ctk.CTkFrame(self, fg_color=C.NIGHT)
        self.content.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._extended_hours: bool = False   # toggled from ResearchPage

        page_defs = [
            ("status",     StatusPage),
            ("trades",     TradesPage),
            ("console",    ConsolePage),
            ("market",     MarketPage),
            ("research",   ResearchPage),
            ("backtests",  BacktestsPage),
            ("strategy",   StrategyPage),
            ("postmortem", PostmortemPage),
            ("system",     SystemPage),
            ("admin",      AdminPage),
        ]

        self.pages: Dict[str, ctk.CTkFrame] = {}

        for key, PageClass in page_defs:
            page = PageClass(self.content, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page

        self._navigate("status")

    def _build_statusbar(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=C.VOID, height=28, corner_radius=0)
        bar.grid(row=1, column=1, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        self.toast_lbl = ctk.CTkLabel(
            bar, text=f"{G.DOT_DIM}  ready.", anchor="w",
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_SANS[0], size=11),
        )
        self.toast_lbl.grid(row=0, column=1, padx=14, sticky="w")
        self.clock_lbl = ctk.CTkLabel(
            bar, text="",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
        )
        self.clock_lbl.grid(row=0, column=2, padx=14, sticky="e")

        # 覇者 = HASHA — "conqueror"
        self.watermark = ctk.CTkLabel(
            bar,
            text="  覇者  ",
            text_color=C.BRUSH_RED,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=13, weight="bold"),
        )
        self.watermark.grid(row=0, column=0, padx=(8, 0), sticky="w")

        self._tick_clock()

        # Bottom-left of sidebar: drawn watermark of Ryo Narushima
        # (Shamo) — hooded figure ringed by bird skulls.
        self._build_corner_watermark()

    def _build_corner_watermark(self) -> None:
        """覇者 (conqueror) + BORN TO CONQUER caption — large kanji,
        no helmet, pure canvas drawing."""
        import tkinter as tk

        sidebar = None
        for child in self.grid_slaves():
            info = child.grid_info()
            if info.get("column") == 0 and info.get("row") == 0:
                sidebar = child
                break
        if sidebar is None:
            return

        wm = ctk.CTkFrame(sidebar, fg_color="transparent")
        wm.grid(row=101, column=0, padx=10, pady=(0, 20), sticky="sw")
        sidebar.grid_rowconfigure(101, weight=0)

        cw, ch = 180, 220
        c = tk.Canvas(wm, width=cw, height=ch,
                       bg=C.VOID, highlightthickness=0)
        c.pack(anchor="w")

        cx = cw / 2
        BLOOD = C.BRUSH_RED

        # 覇者 (HASHA — "conqueror") stacked vertically, very large
        kanji_size = 64
        for i, ch_char in enumerate("覇者"):
            y = 50 + i * (kanji_size + 8)
            # Subtle shadow for depth
            c.create_text(
                cx + 2, y + 2,
                text=ch_char, fill="#1a0408",
                font=(FONT_DISPLAY[0], kanji_size, "bold"),
            )
            # Main glyph
            c.create_text(
                cx, y,
                text=ch_char, fill=BLOOD,
                font=(FONT_DISPLAY[0], kanji_size, "bold"),
            )

        # English subtitle
        c.create_text(
            cx, ch - 18, text="BORN TO CONQUER",
            fill=C.PARCHMENT,
            font=(FONT_SANS[0], 10, "bold"),
        )
        c.create_text(
            cx, ch - 5, text="✠  ✠  ✠",
            fill=C.SCAR,
            font=(FONT_SANS[0], 9),
        )

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _navigate(self, key: str) -> None:
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=C.BLOOD_DIM, text_color=C.BONE)
            else:
                btn.configure(fg_color="transparent", text_color=C.PARCHMENT)
        self.pages[key].tkraise()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def toast(self, message: str) -> None:
        self.toast_lbl.configure(text=f"{G.DOT_ON}  {message}", text_color=C.BONE)
        self.after(5000, lambda: self.toast_lbl.configure(
            text=f"{G.DOT_DIM}  ready.", text_color=C.ASH))

    def _tick_clock(self) -> None:
        self.clock_lbl.configure(
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
        try: self.poller.stop()
        except Exception: pass
        self.destroy()


