"""Market Data — custom-drawn candlestick charts for each tracked symbol.

Uses raw tk.Canvas rather than matplotlib so the visual style matches
the rest of the app — ink-brushed, dark, atmospheric. Data fetched
via WebDataScraper (Yahoo OHLCV, no API key needed).
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk
import pandas as pd

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS
from brzrkr_app.widgets import (
    CodexBox, GhostButton, GothicCard, PageTitle, RuneButton, SectionHeader,
)
from brzrkr_app.theme import FONT_SERIF
from brzrkr_app.indicators import (
    EXPLANATIONS, TIME_SPREADS, all_indicators, bollinger,
)


class MarketPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._current_df: Optional[pd.DataFrame] = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=2)
        self.grid_rowconfigure(5, weight=1)

        PageTitle(self, f"{G.RUNE_R} Market Sight",
                   subtitle="candles · trends · omens").grid(
            row=0, column=0, sticky="ew", pady=(0, 12))

        # Top control bar
        bar = GothicCard(self)
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        bar.grid_columnconfigure(0, weight=1)
        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="ew", padx=14, pady=10)
        inner.grid_columnconfigure(6, weight=1)  # spacer between controls and price

        # ── SYMBOL label + text entry
        ctk.CTkLabel(inner, text="SYMBOL",
                      text_color=C.PARCHMENT,
                      font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold")
                      ).grid(row=0, column=0, padx=(0, 6))
        self.v_symbol = ctk.StringVar(value="SPY")
        ctk.CTkEntry(inner, textvariable=self.v_symbol, width=100,
                      fg_color=C.OBSIDIAN, border_color=C.BORDER,
                      text_color=C.BONE).grid(row=0, column=1)

        # ── Universe dropdown  ────────────────────────────────────────
        # Build sorted label list: "SPY  (INDEX_ETF)" etc.
        try:
            from src.data.market_collector import WIDE_UNIVERSE as _WU
            _sym_labels = sorted(
                [f"{s}  ({t.replace('_',' ')})" for s, t in _WU.items()])
        except Exception:
            _sym_labels = ["SPY  (INDEX ETF)", "QQQ  (INDEX ETF)",
                           "AAPL  (STOCK)", "NVDA  (STOCK)", "TSLA  (STOCK)"]

        def _on_sym_pick(choice: str) -> None:
            sym = choice.split()[0].strip()
            self.v_symbol.set(sym)
            self._show_cached_price(sym)

        self._sym_combo = ctk.CTkComboBox(
            inner,
            values=_sym_labels,
            width=220,
            fg_color=C.OBSIDIAN,
            border_color=C.BORDER,
            button_color=C.BLOOD_DIM,
            button_hover_color=C.BLOOD,
            text_color=C.BONE,
            dropdown_fg_color=C.PANEL,
            dropdown_text_color=C.BONE,
            command=_on_sym_pick,
        )
        self._sym_combo.grid(row=0, column=2, padx=(8, 0))

        # ── TIME SPREAD
        ctk.CTkLabel(inner, text="SPREAD",
                      text_color=C.PARCHMENT,
                      font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold")
                      ).grid(row=0, column=3, padx=(16, 6))
        self.v_spread = ctk.StringVar(value="1 day  — last 6 months")
        ctk.CTkOptionMenu(inner,
                           values=list(TIME_SPREADS.keys()),
                           variable=self.v_spread, width=190,
                           fg_color=C.PANEL_HI,
                           button_color=C.BLOOD_DIM,
                           button_hover_color=C.BLOOD,
                           text_color=C.BONE,
                           dropdown_fg_color=C.PANEL
                           ).grid(row=0, column=4)

        RuneButton(inner, "Scry", glyph=G.RUNE_O,
                    command=self._fetch
                    ).grid(row=0, column=5, padx=(14, 0))

        # ── Live price display  (column 6 = spacer, price at 7)
        self._price_lbl = ctk.CTkLabel(
            inner,
            text="——",
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=20, weight="bold"),
            text_color=C.PARCHMENT,
            anchor="e",
        )
        self._price_lbl.grid(row=0, column=7, padx=(0, 4), sticky="e")
        self._price_chg_lbl = ctk.CTkLabel(
            inner,
            text="",
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
            text_color=C.ASH,
            anchor="e",
        )
        self._price_chg_lbl.grid(row=0, column=8, padx=(0, 0), sticky="e")

        # Show cached price for default symbol on load
        self.after(800, lambda: self._show_cached_price("SPY"))

        # Chart card
        chart_card = GothicCard(self)
        chart_card.grid(row=2, column=0, sticky="nsew")
        chart_card.grid_columnconfigure(0, weight=1)
        chart_card.grid_rowconfigure(1, weight=1)

        self.title_lbl = SectionHeader(chart_card, "Awaiting Sight", glyph=G.MOON)
        self.title_lbl.grid(row=0, column=0, sticky="ew")

        self.canvas_frame = ctk.CTkFrame(chart_card, fg_color=C.VOID)
        self.canvas_frame.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg=C.VOID,
                                 highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # Stats footer
        self.stats_lbl = ctk.CTkLabel(
            chart_card, text="",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
            anchor="w")
        self.stats_lbl.grid(row=2, column=0, padx=14, pady=(0, 12), sticky="ew")

        # ---- Key Indicators panel ----------------------------------
        ind_card = GothicCard(self)
        ind_card.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ind_card.grid_columnconfigure(0, weight=1)
        SectionHeader(ind_card, "Key Market Indicators",
                       glyph=G.RUNE_T).grid(row=0, column=0, sticky="ew")
        self.ind_grid = ctk.CTkFrame(ind_card, fg_color="transparent")
        self.ind_grid.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 14))
        for col in range(3):
            self.ind_grid.grid_columnconfigure(col, weight=1)
        # Built dynamically when data arrives
        self._ind_cells: list[ctk.CTkFrame] = []
        ctk.CTkLabel(
            self.ind_grid,
            text=f"  {G.DOT_DIM}  Scry a symbol above — indicators appear here.",
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, padx=4, pady=8, sticky="w")

        # ---- Indicator Explanations (collapsible) ------------------
        exp_card = GothicCard(self)
        exp_card.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        exp_card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(exp_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        SectionHeader(head, "Indicator Explanations",
                       glyph=G.RUNE_O).grid(row=0, column=0, sticky="w")
        self._exp_open = False
        self._exp_btn = ctk.CTkButton(
            head, text="show / hide", width=110,
            fg_color="transparent",
            hover_color=C.PANEL_HI,
            text_color=C.PARCHMENT,
            border_color=C.BORDER, border_width=1,
            font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
            command=self._toggle_explanations,
        )
        self._exp_btn.grid(row=0, column=1, padx=(0, 14), pady=(10, 4),
                            sticky="e")
        self._exp_body = CodexBox(exp_card, height=320)
        # Build explanation text once (static)
        lines = []
        for name, info in EXPLANATIONS.items():
            lines.append(f"  {G.CROSS}  {name}")
            lines.append(f"  ──────────────────────────────────────────────────────")
            lines.append(f"  WHAT:       {info['what']}")
            lines.append(f"  FORMULA:    {info['formula']}")
            lines.append(f"  SIGNALS:    {info['signals']}")
            lines.append(f"  PITFALL:    {info['pitfall']}")
            lines.append("")
        self._exp_body.set_text("\n".join(lines))
        # Hidden by default; revealed by _toggle_explanations.

        # ---- Sweep Scanner panel --------------------------------------
        sweep_card = GothicCard(self)
        sweep_card.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        sweep_card.grid_columnconfigure(0, weight=1)
        sweep_card.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(sweep_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        SectionHeader(head, "Sweep Scanner  (small / penny stocks)",
                       glyph=G.SKULL).grid(row=0, column=0, sticky="w")
        RuneButton(head, "Scan Now", glyph=G.RUNE_F,
                    command=self._scan_sweeps
                    ).grid(row=0, column=1, padx=(0, 14), pady=(10, 4),
                            sticky="e")

        ctk.CTkLabel(
            sweep_card,
            text=(f"  {G.RIGHT}  Flags symbols where today's volume + price both "
                  f"break the trailing 20-day baseline.\n"
                  f"  {G.RIGHT}  Alerts append to data/sweeps.jsonl. "
                  f"For continuous scanning: python scan_sweeps.py --interval 300"),
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_SERIF[0], size=11),
            justify="left", anchor="w",
        ).grid(row=1, column=0, padx=14, pady=(0, 6), sticky="ew")

        self.sweep_box = CodexBox(sweep_card, height=200)
        self.sweep_box.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self._refresh_sweeps()

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
        # Market data updates independently; nothing tied to broker snap.
        pass

    # ------------------------------------------------------------------
    def _show_cached_price(self, symbol: str) -> None:
        """Show price from VectorStore cache — instant, no network call."""
        try:
            from src.data.vector_store import VectorStore
            snaps = VectorStore().load_latest()
            for s in snaps:
                if s.get("symbol") == symbol:
                    price = s.get("price", 0)
                    chg   = s.get("change_1d_pct", 0)
                    arrow = "▲" if chg >= 0 else "▼"
                    chg_color = C.LIFE if chg >= 0 else C.DEATH
                    self._price_lbl.configure(
                        text=f"${price:,.2f}",
                        text_color=chg_color,
                    )
                    self._price_chg_lbl.configure(
                        text=f"{arrow} {abs(chg):.2f}%  1d",
                        text_color=chg_color,
                    )
                    return
        except Exception:
            pass
        # Not in cache yet — show dashes
        self._price_lbl.configure(text="——", text_color=C.PARCHMENT)
        self._price_chg_lbl.configure(text="")

    def _update_price_from_df(self, df: "pd.DataFrame") -> None:
        """Update price label from freshly fetched OHLCV dataframe."""
        try:
            close = df["close"].astype(float)
            price = float(close.iloc[-1])
            chg   = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0
            arrow = "▲" if chg >= 0 else "▼"
            chg_color = C.LIFE if chg >= 0 else C.DEATH
            self._price_lbl.configure(
                text=f"${price:,.2f}",
                text_color=chg_color,
            )
            self._price_chg_lbl.configure(
                text=f"{arrow} {abs(chg):.2f}%  1d",
                text_color=chg_color,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _fetch(self) -> None:
        symbol = self.v_symbol.get().strip().upper()
        spread = self.v_spread.get()
        if not symbol:
            self.app.toast("A name is required.")
            return
        if spread not in TIME_SPREADS:
            self.app.toast("Pick a valid time spread.")
            return
        rng, interval = TIME_SPREADS[spread]
        self.app.toast(f"Scrying {symbol} ({spread.strip()})…")
        threading.Thread(target=self._fetch_thread,
                          args=(symbol, spread, rng, interval),
                          daemon=True).start()

    def _fetch_thread(self, symbol: str, spread: str,
                       rng: str, interval: str) -> None:
        try:
            from src.data_scraper import WebDataScraper
            agent = WebDataScraper()
            df = agent.scrape_ohlcv(symbol, range_=rng, interval=interval)
        except Exception as exc:  # noqa: BLE001
            try:
                self.after(0, lambda e=exc: self.app.toast(f"Scry failed: {e}"))
            except RuntimeError:
                pass
            return
        try:
            self.after(0, lambda d=df, s=symbol, sp=spread:
                        self._on_data(d, s, sp))
        except RuntimeError:
            pass

    def _on_data(self, df: pd.DataFrame, symbol: str, spread: str) -> None:
        if df is None or df.empty:
            self.app.toast(f"No data for {symbol}")
            return
        self._current_df = df
        self._current_symbol = symbol
        self.title_lbl._label.configure(
            text=f"{G.SIGIL_L}  {G.RUNE_R}  {symbol}  ·  {spread}  {G.RUNE_R}  {G.SIGIL_R}"
        )
        self._update_price_from_df(df)
        last = float(df["close"].iloc[-1])
        first = float(df["close"].iloc[0])
        chg = (last / first - 1.0) * 100.0
        hi = float(df["high"].max())
        lo = float(df["low"].min())
        self.stats_lbl.configure(
            text=(f"  last ${last:>9,.2f}     chg {chg:+.2f}%     "
                  f"hi ${hi:>9,.2f}     lo ${lo:>9,.2f}     "
                  f"bars {len(df):>4d}"),
            text_color=C.LIFE if chg >= 0 else C.DEATH,
        )
        self._redraw()
        self._refresh_indicators(df)

    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        if self._current_df is None or self._current_df.empty:
            w = c.winfo_width() or 800
            h = c.winfo_height() or 400
            c.create_text(w // 2, h // 2,
                          text=f"{G.MOON}  no sight yet  {G.MOON}",
                          fill=C.ASH,
                          font=(FONT_DISPLAY[0], 18, "bold"))
            return

        df = self._current_df.tail(260)  # ~1y cap
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or h < 20:
            return

        # Margins
        ml, mr, mt, mb = 60, 20, 20, 30
        plot_w = w - ml - mr
        plot_h = h - mt - mb

        # Price scale
        hi = float(df["high"].max())
        lo = float(df["low"].min())
        if hi == lo:
            hi += 1
        pad = (hi - lo) * 0.05
        hi += pad
        lo -= pad

        def y(price):
            return mt + plot_h - (price - lo) / (hi - lo) * plot_h

        # Background grid
        for i in range(6):
            yv = mt + plot_h * i / 5
            price = hi - (hi - lo) * i / 5
            c.create_line(ml, yv, ml + plot_w, yv, fill=C.IRON, width=1)
            c.create_text(ml - 6, yv, anchor="e",
                          text=f"${price:,.1f}",
                          fill=C.ASH,
                          font=(FONT_MONO[0], 9))

        # Candlestick widths
        n = len(df)
        bar_w = max(1.0, plot_w / n * 0.7)
        slot = plot_w / max(n, 1)

        for i, (ts, row) in enumerate(df.iterrows()):
            x = ml + (i + 0.5) * slot
            o = float(row["open"]); cl = float(row["close"])
            hi_p = float(row["high"]); lo_p = float(row["low"])
            up = cl >= o
            color = C.LIFE if up else C.DEATH

            # Wick
            c.create_line(x, y(hi_p), x, y(lo_p), fill=color, width=1)
            # Body
            top = y(max(o, cl))
            bot = y(min(o, cl))
            if abs(bot - top) < 1:
                bot = top + 1
            c.create_rectangle(x - bar_w / 2, top, x + bar_w / 2, bot,
                                fill=color, outline=color)

        # Border around plot
        c.create_rectangle(ml, mt, ml + plot_w, mt + plot_h,
                            outline=C.BORDER, width=1)

        # ---- Overlays: SMA20 / SMA50 + Bollinger Bands -----------
        if len(df) >= 20:
            sma20 = df["close"].rolling(20).mean()
            self._draw_line_overlay(c, sma20, ml, mt, plot_w, plot_h,
                                      slot, y, color=C.OMEN, width=1)
        if len(df) >= 50:
            sma50 = df["close"].rolling(50).mean()
            self._draw_line_overlay(c, sma50, ml, mt, plot_w, plot_h,
                                      slot, y, color=C.WIND, width=1)
        if len(df) >= 20:
            upper, mid, lower = bollinger(df["close"])
            self._draw_line_overlay(c, upper, ml, mt, plot_w, plot_h,
                                      slot, y, color=C.SCAR, width=1,
                                      dash=(2, 3))
            self._draw_line_overlay(c, lower, ml, mt, plot_w, plot_h,
                                      slot, y, color=C.SCAR, width=1,
                                      dash=(2, 3))

        # Legend
        legend_y = mt + 8
        for i, (label, col) in enumerate([
            ("SMA20", C.OMEN), ("SMA50", C.WIND),
            ("BB(20,2)", C.SCAR),
        ]):
            c.create_text(ml + plot_w - 8 - i * 80, legend_y,
                           text=label, anchor="e",
                           fill=col,
                           font=(FONT_MONO[0], 9, "bold"))

        # Date ticks
        for i in (0, n // 4, n // 2, 3 * n // 4, n - 1):
            if 0 <= i < n:
                x = ml + (i + 0.5) * slot
                ts = df.index[i]
                c.create_text(x, mt + plot_h + 12,
                              text=ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "hour") else ts.strftime("%Y-%m-%d"),
                              fill=C.ASH,
                              font=(FONT_MONO[0], 9))

    def _draw_line_overlay(self, c, series, ml: float, mt: float,
                            plot_w: float, plot_h: float, slot: float,
                            y_fn, *, color: str, width: int = 1,
                            dash=None) -> None:
        """Draw a line series over the candle chart, skipping NaN segments."""
        pts = []
        for i, v in enumerate(series.values):
            if pd.isna(v):
                if len(pts) >= 4:
                    if dash:
                        c.create_line(*pts, fill=color, width=width, dash=dash)
                    else:
                        c.create_line(*pts, fill=color, width=width,
                                        smooth=False)
                pts = []
                continue
            x = ml + (i + 0.5) * slot
            pts.extend([x, y_fn(v)])
        if len(pts) >= 4:
            if dash:
                c.create_line(*pts, fill=color, width=width, dash=dash)
            else:
                c.create_line(*pts, fill=color, width=width, smooth=False)

    # ------------------------------------------------------------------
    # Sweep scanner
    # ------------------------------------------------------------------
    def _refresh_sweeps(self) -> None:
        try:
            from src.scanners.sweep_detector import SweepDetector
            recent = SweepDetector.load_recent(limit=20)
            if not recent:
                self.sweep_box.set_text(
                    f"  {G.DOT_DIM}  no sweep alerts yet.\n\n"
                    f"  Click 'Scan Now' or run from CLI:\n"
                    f"     python scan_sweeps.py                # one-shot\n"
                    f"     python scan_sweeps.py --interval 300 # every 5 min")
                return
            lines = []
            for a in recent:
                arrow = "↑" if a.get("direction") == "up" else "↓"
                t = (a.get("detected_at") or "")[:16].replace("T", " ")
                lines.append(
                    f"  {arrow}  {a.get('symbol','?'):<6s}  "
                    f"${a.get('close',0):>8.2f}  "
                    f"{a.get('price_change_pct',0):+6.2f}%   "
                    f"vol {a.get('volume_ratio',0):>5.2f}×   "
                    f"σ {a.get('realized_vol_sigmas',0):>4.2f}   "
                    f"score {a.get('score',0):>6.2f}   "
                    f"{t}"
                )
            self.sweep_box.set_text("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            self.sweep_box.set_text(f"  {G.SKULL}  read failed: {exc}")

    def _scan_sweeps(self) -> None:
        self.app.toast(f"{G.RUNE_F}  Scanning small caps... (~30s)")
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _scan_thread(self) -> None:
        try:
            from src.scanners.sweep_detector import SweepDetector
            d = SweepDetector()
            alerts = d.scan()
            if alerts:
                d.write(alerts)
            try:
                self.after(0, lambda n=len(alerts): (
                    self.app.toast(f"Sweep scan done — {n} alert(s)."),
                    self._refresh_sweeps(),
                ))
            except RuntimeError:
                pass
        except Exception as exc:  # noqa: BLE001
            try:
                self.after(0, lambda e=exc: self.app.toast(f"Scan failed: {e}"))
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # Indicators panel
    # ------------------------------------------------------------------
    def _refresh_indicators(self, df: pd.DataFrame) -> None:
        # Clear existing cells
        for cell in self._ind_cells:
            cell.destroy()
        for child in self.ind_grid.winfo_children():
            child.destroy()
        self._ind_cells = []

        readings = all_indicators(df)
        STATE_COLORS = {
            "bullish": C.LIFE,
            "bearish": C.WOUND,
            "neutral": C.PARCHMENT,
        }
        STATE_DOT = {
            "bullish": "●",
            "bearish": "●",
            "neutral": "○",
        }

        for i, r in enumerate(readings):
            row, col = divmod(i, 3)
            color = STATE_COLORS.get(r.state, C.PARCHMENT)
            cell = ctk.CTkFrame(
                self.ind_grid, fg_color=C.PANEL,
                border_color=C.BORDER, border_width=1,
                corner_radius=2,
            )
            cell.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            cell.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                cell, text=r.name.upper(),
                text_color=C.ASH,
                font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, padx=12, pady=(10, 2), sticky="ew")
            ctk.CTkLabel(
                cell,
                text=f"{STATE_DOT[r.state]}  {r.display}",
                text_color=color,
                font=ctk.CTkFont(family=FONT_MONO[0], size=18, weight="bold"),
                anchor="w",
            ).grid(row=1, column=0, padx=12, pady=0, sticky="ew")
            ctk.CTkLabel(
                cell, text=r.note,
                text_color=C.PARCHMENT,
                font=ctk.CTkFont(family=FONT_SANS[0], size=10),
                anchor="w", justify="left", wraplength=240,
            ).grid(row=2, column=0, padx=12, pady=(2, 10), sticky="ew")
            self._ind_cells.append(cell)

    # ------------------------------------------------------------------
    # Explanations toggle
    # ------------------------------------------------------------------
    def _toggle_explanations(self) -> None:
        self._exp_open = not self._exp_open
        if self._exp_open:
            self._exp_body.grid(row=1, column=0, padx=14, pady=(0, 14),
                                  sticky="ew")
        else:
            self._exp_body.grid_remove()
