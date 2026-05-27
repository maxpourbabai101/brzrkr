"""Research — Market Intelligence dashboard.

Reads from the VectorStore cache (populated by BackgroundCollector) for
near-instant display.  Falls back to a live yfinance scan if the cache
is empty or stale.

Layout (top→bottom)
──────────────────
  Row 0 — PageTitle
  Row 1 — Status strip  (Regime / Session / Edge / Collector heartbeat)
  Row 2 — Power Plays   (Futures | Options | Swing — top picks per category)
  Row 3 — Universe Watchlist  (full scored table, filter bar)
  Row 4 — Signal Intelligence log  (last 20 signals with outcomes)
  Row 5 — Learning Feed  (model updates, journal stats, lessons)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, pnl_color
from brzrkr_app.widgets import BloodMetric, CodexBox, GothicCard, PageTitle, SectionHeader

# ── Paths ─────────────────────────────────────────────────────────────
_SIGNAL_DIR   = Path("data/signals")
_REGIME_CACHE = Path("data/regime_cache.json")
_LEARN_REPORT = Path("data/learning_report.json")
_JOURNAL_PATH = Path("data/trade_journal.jsonl")


# ── Helpers ───────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None


def _session_now() -> Dict[str, str]:
    import zoneinfo
    et  = zoneinfo.ZoneInfo("America/New_York")
    now = datetime.now(et)
    h   = now.hour + now.minute / 60.0
    if 4.0 <= h < 9.5:
        label, color, nxt = "PRE-MARKET",  C.OMEN,   "Regular opens 09:30 ET"
    elif 9.5 <= h < 16.0:
        label, color, nxt = "REGULAR",     C.LIFE,   "After-hours opens 16:00 ET"
    elif 16.0 <= h < 20.0:
        label, color, nxt = "AFTER-HOURS", C.OMEN,   "Market closed 20:00 ET"
    else:
        label, color, nxt = "CLOSED",      C.GHOST,  "Pre-market opens 04:00 ET"
    return {"label": label, "color": color, "next": nxt,
            "time": now.strftime("%H:%M ET")}


def _regime_color(label: str) -> str:
    return {"trending_up": C.LIFE, "trending_down": C.DEATH,
            "ranging": C.PARCHMENT, "volatile": C.WOUND}.get(label, C.ASH)


# ══════════════════════════════════════════════════════════════════════
# ResearchPage
# ══════════════════════════════════════════════════════════════════════

class ResearchPage(ctk.CTkFrame):
    # How often to re-render power plays from existing cache data
    PP_REFRESH_MS   = 3_000
    # How often to reload latest.json from disk (low cost)
    CACHE_RELOAD_MS = 15_000
    # Force a fresh yfinance scan when cache is this stale (minutes)
    STALE_MINUTES   = 25

    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app            = app
        self._snapshots:    List[Dict] = []   # current from VectorStore
        self._scan_running  = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # watchlist expands

        PageTitle(
            self, f"{G.RUNE_O} Research",
            subtitle="market intelligence · power plays · learning feed",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self._build_status_strip(row=1)
        self._build_power_plays(row=2)
        self._build_watchlist(row=3)
        self._build_bottom_row(row=4)

        # Start data load cycle
        self.after(300,  self._reload_cache)
        self.after(500,  self._refresh_status_strip)
        self.after(8_000, self._tick_status)

    # ──────────────────────────────────────────────────────────────────
    # ① STATUS STRIP
    # ──────────────────────────────────────────────────────────────────

    def _build_status_strip(self, row: int) -> None:
        strip = ctk.CTkFrame(self, fg_color="transparent")
        strip.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        for i in range(4):
            strip.grid_columnconfigure(i, weight=1)

        # ── Regime ────────────────────────────────────────────────────
        rc = GothicCard(strip)
        rc.grid(row=0, column=0, padx=(0, 4), sticky="nsew")
        rc.grid_columnconfigure(0, weight=1)
        SectionHeader(rc, "Market Regime", glyph=G.RUNE_T).grid(
            row=0, column=0, sticky="ew")
        self._regime_lbl = ctk.CTkLabel(
            rc, text="——",
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=18, weight="bold"),
            text_color=C.PARCHMENT, anchor="w")
        self._regime_lbl.grid(row=1, column=0, padx=14, pady=(4, 0), sticky="ew")
        self._regime_detail = ctk.CTkLabel(
            rc, text="loading…", text_color=C.ASH, anchor="w", justify="left",
            font=ctk.CTkFont(family=FONT_MONO[0], size=9), wraplength=190)
        self._regime_detail.grid(row=2, column=0, padx=14, pady=(2, 10), sticky="ew")

        # ── Session ───────────────────────────────────────────────────
        sc = GothicCard(strip)
        sc.grid(row=0, column=1, padx=4, sticky="nsew")
        sc.grid_columnconfigure(0, weight=1)
        SectionHeader(sc, "Session", glyph=G.MOON).grid(row=0, column=0, sticky="ew")
        self._session_lbl = ctk.CTkLabel(
            sc, text="——",
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=18, weight="bold"),
            text_color=C.PARCHMENT, anchor="w")
        self._session_lbl.grid(row=1, column=0, padx=14, pady=(4, 0), sticky="ew")
        self._session_next = ctk.CTkLabel(sc, text="", text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_SANS[0], size=9), anchor="w")
        self._session_next.grid(row=2, column=0, padx=14, pady=(2, 4), sticky="ew")
        self.v_ext_hours = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            sc, text="Extended-hours trading",
            variable=self.v_ext_hours,
            fg_color=C.OMEN, hover_color=C.BLOOD,
            border_color=C.BORDER, text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_SANS[0], size=9, weight="bold"),
            command=self._on_ext_toggle,
        ).grid(row=3, column=0, padx=14, pady=(0, 10), sticky="w")

        # ── Model Edge ────────────────────────────────────────────────
        ec = GothicCard(strip)
        ec.grid(row=0, column=2, padx=4, sticky="nsew")
        ec.grid_columnconfigure(0, weight=1)
        SectionHeader(ec, "Model Edge", glyph=G.RUNE_F).grid(
            row=0, column=0, sticky="ew")
        self.m_winrate = BloodMetric(ec, "Win Rate",   "——")
        self.m_avgr    = BloodMetric(ec, "Avg R",      "——")
        self.m_conf    = BloodMetric(ec, "Conf. Thr.", "——")
        self.m_winrate.grid(row=1, column=0, padx=10, pady=(2, 0), sticky="ew")
        self.m_avgr.grid(   row=2, column=0, padx=10, pady=1,       sticky="ew")
        self.m_conf.grid(   row=3, column=0, padx=10, pady=(1, 10), sticky="ew")

        # ── Collector Heartbeat ───────────────────────────────────────
        hc = GothicCard(strip)
        hc.grid(row=0, column=3, padx=(4, 0), sticky="nsew")
        hc.grid_columnconfigure(0, weight=1)
        SectionHeader(hc, "Data Collector", glyph=G.GEAR).grid(
            row=0, column=0, sticky="ew")
        self._coll_lbl = ctk.CTkLabel(
            hc, text="——", text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=14, weight="bold"),
            anchor="w")
        self._coll_lbl.grid(row=1, column=0, padx=14, pady=(4, 0), sticky="ew")
        self._coll_detail = ctk.CTkLabel(
            hc, text="Starting…", text_color=C.ASH, anchor="w", justify="left",
            font=ctk.CTkFont(family=FONT_MONO[0], size=9), wraplength=175)
        self._coll_detail.grid(row=2, column=0, padx=14, pady=(2, 4), sticky="ew")
        self._force_scan_btn = ctk.CTkButton(
            hc, text="⟳  Scan Now", height=24, width=90,
            fg_color=C.BLOOD_DIM, hover_color=C.BLOOD,
            text_color=C.BONE, font=ctk.CTkFont(family=FONT_SANS[0], size=9),
            command=self._force_scan,
        )
        self._force_scan_btn.grid(row=3, column=0, padx=14, pady=(0, 10), sticky="w")

    # ──────────────────────────────────────────────────────────────────
    # ② POWER PLAYS
    # ──────────────────────────────────────────────────────────────────

    def _build_power_plays(self, row: int) -> None:
        outer = GothicCard(self)
        outer.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        outer.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(outer, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        SectionHeader(hdr, "Power Plays — Top Setups Per Category",
                       glyph=G.RUNE_F).grid(row=0, column=0, sticky="w")
        self._pp_ts = ctk.CTkLabel(
            hdr, text="⟳ updates every 3s",
            text_color=C.GHOST,
            font=ctk.CTkFont(family=FONT_MONO[0], size=8))
        self._pp_ts.grid(row=0, column=1, padx=(0, 14), pady=(10, 4), sticky="e")

        # Single horizontal row — 4 equal columns, all visible without scrolling
        cards_row = ctk.CTkFrame(outer, fg_color="transparent")
        cards_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        for i in range(4):
            cards_row.grid_columnconfigure(i, weight=1)

        def _make_pp_card(parent, col, title, glyph, subtitle):
            c = GothicCard(parent)
            if col == 0:
                px = (0, 4)
            elif col == 3:
                px = (4, 0)
            else:
                px = (4, 4)
            c.grid(row=0, column=col, padx=px, sticky="nsew")
            c.grid_columnconfigure(0, weight=1)
            SectionHeader(c, title, glyph=glyph).grid(row=0, column=0, sticky="ew")
            ctk.CTkLabel(c, text=subtitle, text_color=C.GHOST,
                          font=ctk.CTkFont(family=FONT_SANS[0], size=8),
                          anchor="w").grid(row=1, column=0, padx=10, pady=(0, 2), sticky="ew")
            lst = ctk.CTkFrame(c, fg_color="transparent")
            lst.grid(row=2, column=0, padx=6, pady=(0, 4), sticky="ew")
            lst.grid_columnconfigure(0, weight=1)
            ts = ctk.CTkLabel(c, text="—", text_color=C.IRON,
                               font=ctk.CTkFont(family=FONT_MONO[0], size=7), anchor="w")
            ts.grid(row=3, column=0, padx=10, pady=(0, 5), sticky="ew")
            return lst, ts

        self._pp_futures_lst, self._pp_futures_ts = _make_pp_card(
            cards_row, 0, "FUTURES", G.RUNE_T,
            "commodity · bond · leveraged")
        self._pp_options_lst, self._pp_options_ts = _make_pp_card(
            cards_row, 1, "OPTIONS", G.EXEC,
            "vol surge · RSI extreme")
        self._pp_swing_lst, self._pp_swing_ts = _make_pp_card(
            cards_row, 2, "SWING", G.RUNE_O,
            "stocks · MACD · momentum")
        self._pp_penny_lst, self._pp_penny_ts = _make_pp_card(
            cards_row, 3, "PENNY", G.SKULL,
            "price < $5 · vol surge")

        # Start 3s refresh loop
        self.after(self.PP_REFRESH_MS, self._refresh_power_plays)

    def _refresh_power_plays(self) -> None:
        """Rebuild power play cards from cached snapshots."""
        ts_str = datetime.now().strftime("%H:%M:%S")
        self._pp_ts.configure(text=f"⟳ {ts_str}", text_color=C.SIGIL)

        def _clear(frame):
            for w in frame.winfo_children():
                w.destroy()

        def _placeholder(frame, msg):
            _clear(frame)
            ctk.CTkLabel(frame, text=msg, text_color=C.GHOST,
                          font=ctk.CTkFont(family=FONT_MONO[0], size=9),
                          anchor="w").grid(row=0, column=0, sticky="ew", pady=4)

        def _pick_row(frame, row_i, s: dict, reason: str) -> None:
            """Compact 2-line pick row sized for narrow 4-column cards."""
            bullish = s.get("signal", "neutral").lower() == "bullish"
            fg      = C.LIFE if bullish else C.DEATH
            score   = s.get("score", 0)
            price   = s.get("price", 0)
            chg     = s.get("change_1d_pct", 0)
            arrow   = "↑" if chg >= 0 else "↓"
            sign    = "+" if score >= 0 else ""

            row_f = ctk.CTkFrame(frame, fg_color=C.OBSIDIAN, corner_radius=3)
            row_f.grid(row=row_i, column=0, sticky="ew", pady=1)
            row_f.grid_columnconfigure(1, weight=1)

            # Line 1: symbol + score badge
            top = ctk.CTkFrame(row_f, fg_color="transparent")
            top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(3, 0))
            top.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(top, text=s.get("symbol", "?"),
                          font=ctk.CTkFont(family=FONT_MONO[0], size=10, weight="bold"),
                          text_color=fg, anchor="w",
                          ).pack(side="left")
            ctk.CTkLabel(top,
                          text=f" {sign}{score:.0f}",
                          font=ctk.CTkFont(family=FONT_MONO[0], size=9, weight="bold"),
                          text_color=C.BLOOD_HI if bullish else C.WOUND,
                          anchor="w",
                          ).pack(side="left", padx=(4, 0))
            ctk.CTkLabel(top,
                          text=f"${price:,.2f} {arrow}{abs(chg):.1f}%",
                          font=ctk.CTkFont(family=FONT_MONO[0], size=9),
                          text_color=C.ASH, anchor="e",
                          ).pack(side="right")

            # Line 2: reason (truncated to fit)
            ctk.CTkLabel(row_f,
                          text=reason[:48],
                          font=ctk.CTkFont(family=FONT_SANS[0], size=8),
                          text_color=C.PARCHMENT, anchor="w",
                          ).grid(row=1, column=0, columnspan=2,
                                 padx=5, pady=(0, 3), sticky="ew")

        if not self._snapshots:
            msg = "Collecting data…  (~20s first run)"
            _placeholder(self._pp_futures_lst, msg)
            _placeholder(self._pp_options_lst, msg)
            _placeholder(self._pp_swing_lst,   msg)
            _placeholder(self._pp_penny_lst,   msg)
            self.after(self.PP_REFRESH_MS, self._refresh_power_plays)
            return

        ts_now = datetime.now().strftime("%H:%M:%S")

        # ── FUTURES picks ─────────────────────────────────────────────
        _FUT = {"COMMODITY_ETF", "BOND_ETF", "LEVERAGED_ETF",
                "VOLATILITY", "INDEX_ETF"}
        fut_picks = sorted(
            [s for s in self._snapshots if s.get("asset_type") in _FUT],
            key=lambda s: abs(s.get("score", 0)), reverse=True)[:4]
        _clear(self._pp_futures_lst)
        for i, s in enumerate(fut_picks):
            vr = s.get("vol_ratio", 1)
            vol_note = f"vol {vr:.1f}×" if vr > 1.5 else ""
            reason = "  ·  ".join(filter(None, [s.get("note",""), vol_note]))[:60]
            _pick_row(self._pp_futures_lst, i, s, reason or "—")
        if not fut_picks:
            _placeholder(self._pp_futures_lst, "No futures proxies with signal")
        self._pp_futures_ts.configure(text=f"updated {ts_now}")

        # ── OPTIONS picks ─────────────────────────────────────────────
        def _opts_key(s):
            rsi = s.get("rsi", 50)
            vr  = s.get("vol_ratio", 1)
            return abs(s.get("score", 0)) + abs(rsi - 50) * 0.6 + (vr - 1) * 8

        opt_picks = sorted(
            [s for s in self._snapshots
             if s.get("vol_ratio", 1) > 1.4
             or s.get("rsi", 50) < 35 or s.get("rsi", 50) > 65],
            key=_opts_key, reverse=True)[:4]
        _clear(self._pp_options_lst)
        for i, s in enumerate(opt_picks):
            rsi = s.get("rsi", 50)
            rsi_lbl = ("oversold" if rsi < 35 else "overbought" if rsi > 65 else "neutral")
            vr = s.get("vol_ratio", 1)
            reason = f"RSI {rsi:.0f} {rsi_lbl}  ·  vol {vr:.1f}×"
            _pick_row(self._pp_options_lst, i, s, reason)
        if not opt_picks:
            _placeholder(self._pp_options_lst, "No options setups yet")
        self._pp_options_ts.configure(text=f"updated {ts_now}")

        # ── SWING picks ───────────────────────────────────────────────
        _SWG = {"STOCK", "INDEX_ETF", "SECTOR_ETF"}
        swing_picks = sorted(
            [s for s in self._snapshots
             if s.get("asset_type") in _SWG and abs(s.get("momentum_5d", 0)) > 0.8],
            key=lambda s: abs(s.get("score", 0)), reverse=True)[:4]
        _clear(self._pp_swing_lst)
        for i, s in enumerate(swing_picks):
            mom = s.get("momentum_5d", 0)
            macd = s.get("macd_signal", "flat")
            trend = s.get("trend", "flat")
            macd_lbl = {"cross_up": "MACD ↑", "cross_down": "MACD ↓",
                        "flat": "MACD flat"}.get(macd, macd)
            reason = f"{mom:+.1f}% 5d  ·  {macd_lbl}  ·  {trend}"
            _pick_row(self._pp_swing_lst, i, s, reason)
        if not swing_picks:
            _placeholder(self._pp_swing_lst, "No swing setups yet")
        self._pp_swing_ts.configure(text=f"updated {ts_now}")

        # ── PENNY STOCKS picks ────────────────────────────────────────
        # Penny stocks: price < $5, OR SMALL_CAP type, sorted by vol surge + score
        penny_picks = sorted(
            [s for s in self._snapshots
             if s.get("price", 999) < 5.0
             or s.get("asset_type") == "SMALL_CAP"],
            key=lambda s: (s.get("vol_ratio", 1) * 0.6 + abs(s.get("score", 0)) * 0.4),
            reverse=True,
        )[:4]
        _clear(self._pp_penny_lst)
        for i, s in enumerate(penny_picks):
            price = s.get("price", 0)
            vr    = s.get("vol_ratio", 1)
            rsi   = s.get("rsi", 50)
            mom5  = s.get("momentum_5d", 0)
            reason = f"${price:.2f}  vol {vr:.1f}×  RSI {rsi:.0f}  {mom5:+.1f}% 5d"
            _pick_row(self._pp_penny_lst, i, s, reason)
        if not penny_picks:
            _placeholder(self._pp_penny_lst, "No penny/small-cap picks yet")
        self._pp_penny_ts.configure(text=f"updated {ts_now}")

        self.after(self.PP_REFRESH_MS, self._refresh_power_plays)

    # ──────────────────────────────────────────────────────────────────
    # ③ WATCHLIST
    # ──────────────────────────────────────────────────────────────────

    def _build_watchlist(self, row: int) -> None:
        card = GothicCard(self)
        card.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        SectionHeader(head, "Universe Watchlist — 200+ symbols scored",
                       glyph=G.CROSS).grid(row=0, column=0, sticky="w")
        self._wl_count = ctk.CTkLabel(
            head, text="",
            text_color=C.GHOST,
            font=ctk.CTkFont(family=FONT_MONO[0], size=9))
        self._wl_count.grid(row=0, column=1, padx=(0, 14), pady=(10, 4), sticky="e")

        # Filter bar
        fb = ctk.CTkFrame(card, fg_color="transparent")
        fb.grid(row=1, column=0, padx=14, pady=(0, 4), sticky="ew")
        ctk.CTkLabel(fb, text="FILTER:",
                      text_color=C.ASH,
                      font=ctk.CTkFont(family=FONT_SANS[0], size=9, weight="bold"),
                      ).pack(side="left", padx=(0, 6))
        self.v_filter = ctk.StringVar(value="ALL")
        for lbl in ("ALL", "BULLISH", "BEARISH", "INDEX_ETF", "STOCK",
                     "COMMODITY_ETF", "LEVERAGED_ETF", "SMALL_CAP"):
            ctk.CTkRadioButton(
                fb, text=lbl, variable=self.v_filter, value=lbl,
                fg_color=C.BLOOD, hover_color=C.BLOOD_HI,
                border_color=C.BORDER, text_color=C.ASH,
                font=ctk.CTkFont(family=FONT_SANS[0], size=9),
                command=self._apply_filter,
            ).pack(side="left", padx=3)

        # Treeview
        self.watch_tree = self._make_watch_tree(card)
        self.watch_tree.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="nsew")

    def _make_watch_tree(self, parent) -> ttk.Treeview:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Res.Treeview",
                         background=C.OBSIDIAN, foreground=C.BONE,
                         fieldbackground=C.OBSIDIAN, borderwidth=0,
                         rowheight=20, font=(FONT_MONO[0], 10))
        style.configure("Res.Treeview.Heading",
                         background=C.PANEL_HI, foreground=C.SIGIL,
                         font=(FONT_SANS[0], 9, "bold"), relief="flat")
        style.map("Res.Treeview",
                   background=[("selected", C.BLOOD_DIM)],
                   foreground=[("selected", C.BONE)])
        cols = ("symbol", "type", "score", "signal", "price",
                "chg_1d", "rsi", "macd", "mom_5d", "vol_ratio", "note")
        tree = ttk.Treeview(parent, columns=cols, show="headings",
                             selectmode="browse", style="Res.Treeview", height=12)
        widths = {"symbol": 64, "type": 100, "score": 56, "signal": 72,
                  "price": 78, "chg_1d": 64, "rsi": 50, "macd": 88,
                  "mom_5d": 64, "vol_ratio": 66, "note": 280}
        for c in cols:
            tree.heading(c, text=c.upper().replace("_", " "))
            tree.column(c, anchor="w", width=widths.get(c, 70), stretch=True)
        tree.tag_configure("bullish", foreground=C.LIFE)
        tree.tag_configure("bearish", foreground=C.DEATH)
        tree.tag_configure("neutral", foreground=C.PARCHMENT)
        tree.tag_configure("alert",   foreground=C.OMEN)
        return tree

    def _populate_watchlist(self, snaps: List[Dict]) -> None:
        self.watch_tree.delete(*self.watch_tree.get_children())
        filt = self.v_filter.get()
        shown = 0
        for s in snaps:
            if filt != "ALL":
                if filt in ("BULLISH", "BEARISH"):
                    if s.get("signal", "").upper() != filt:
                        continue
                elif s.get("asset_type") != filt:
                    continue
            score = s.get("score", 0)
            sig   = s.get("signal", "neutral")
            tag   = sig
            if abs(score) > 50:
                tag = "alert"
            macd  = s.get("macd_signal", "flat")
            macd_lbl = {"cross_up": "↑ bull cross", "cross_down": "↓ bear cross",
                        "flat": "—  flat"}.get(macd, macd)
            chg = s.get("change_1d_pct", 0)
            arrow = "↑" if chg >= 0 else "↓"
            self.watch_tree.insert("", "end", tags=(tag,), values=(
                s.get("symbol",""),
                s.get("asset_type","").replace("_", " "),
                f"{score:+.0f}",
                sig.upper(),
                f"${s.get('price',0):,.2f}",
                f"{arrow}{abs(chg):.2f}%",
                f"{s.get('rsi',0):.1f}",
                macd_lbl,
                f"{s.get('momentum_5d',0):+.2f}%",
                f"{s.get('vol_ratio',1):.2f}×",
                s.get("note", ""),
            ))
            shown += 1
        self._wl_count.configure(text=f"{shown} symbols  ·  {len(snaps)} total")

    def _apply_filter(self) -> None:
        if self._snapshots:
            self._populate_watchlist(self._snapshots)

    # ──────────────────────────────────────────────────────────────────
    # ④ BOTTOM ROW: Signal Log + Learning Feed
    # ──────────────────────────────────────────────────────────────────

    def _build_bottom_row(self, row: int) -> None:
        row_f = ctk.CTkFrame(self, fg_color="transparent")
        row_f.grid(row=row, column=0, sticky="ew", pady=(0, 4))
        row_f.grid_columnconfigure(0, weight=3)
        row_f.grid_columnconfigure(1, weight=2)

        # ── Signal Intelligence ───────────────────────────────────────
        sc = GothicCard(row_f)
        sc.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        sc.grid_columnconfigure(0, weight=1)
        sc.grid_rowconfigure(1, weight=1)

        sh = ctk.CTkFrame(sc, fg_color="transparent")
        sh.grid(row=0, column=0, sticky="ew")
        sh.grid_columnconfigure(0, weight=1)
        SectionHeader(sh, "Signal Intelligence — last 20",
                       glyph=G.EXEC).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            sh, text="↻ Reload", width=72, height=22,
            fg_color="transparent", hover_color=C.PANEL_HI,
            text_color=C.PARCHMENT, border_color=C.BORDER, border_width=1,
            font=ctk.CTkFont(family=FONT_SANS[0], size=9),
            command=self._load_signals,
        ).grid(row=0, column=1, padx=(0, 14), pady=(10, 4), sticky="e")

        style = ttk.Style()
        style.configure("Sig.Treeview",
                         background=C.OBSIDIAN, foreground=C.BONE,
                         fieldbackground=C.OBSIDIAN, borderwidth=0,
                         rowheight=19, font=(FONT_MONO[0], 9))
        style.configure("Sig.Treeview.Heading",
                         background=C.PANEL_HI, foreground=C.SIGIL,
                         font=(FONT_SANS[0], 8, "bold"), relief="flat")
        style.map("Sig.Treeview",
                   background=[("selected", C.BLOOD_DIM)],
                   foreground=[("selected", C.BONE)])

        sig_cols = ("time", "symbol", "dir", "conf",
                    "entry", "stop", "target", "outcome")
        self.sig_tree = ttk.Treeview(sc, columns=sig_cols, show="headings",
                                      selectmode="browse",
                                      style="Sig.Treeview", height=6)
        sig_w = {"time": 120, "symbol": 60, "dir": 54, "conf": 50,
                 "entry": 76, "stop": 76, "target": 76, "outcome": 70}
        for c in sig_cols:
            self.sig_tree.heading(c, text=c.upper())
            self.sig_tree.column(c, anchor="w", width=sig_w.get(c, 70), stretch=True)
        self.sig_tree.tag_configure("win",     foreground=C.LIFE)
        self.sig_tree.tag_configure("loss",    foreground=C.DEATH)
        self.sig_tree.tag_configure("open",    foreground=C.OMEN)
        self.sig_tree.tag_configure("dry_run", foreground=C.ASH)
        self.sig_tree.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="nsew")

        # ── Learning Feed ─────────────────────────────────────────────
        lc = GothicCard(row_f)
        lc.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        lc.grid_columnconfigure(0, weight=1)
        SectionHeader(lc, "Learning Feed", glyph=G.SKULL).grid(
            row=0, column=0, sticky="ew")
        self._learn_box = CodexBox(lc, height=180)
        self._learn_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        self._load_signals()
        self._refresh_learning_feed()

    # ──────────────────────────────────────────────────────────────────
    # Data refresh loops
    # ──────────────────────────────────────────────────────────────────

    def update_from(self, snap: dict) -> None:
        pass  # research drives its own refresh cycles

    def _tick_status(self) -> None:
        self._refresh_status_strip()
        self.after(8_000, self._tick_status)

    def _refresh_status_strip(self) -> None:
        # Session
        sess = _session_now()
        self._session_lbl.configure(text=sess["label"], text_color=sess["color"])
        self._session_next.configure(text=f"{sess['time']}  ·  {sess['next']}")

        # Regime
        rd = _load_json(_REGIME_CACHE)
        if rd:
            label = rd.get("label", "unknown")
            self._regime_lbl.configure(
                text=label.replace("_", " ").upper(),
                text_color=_regime_color(label))
            self._regime_detail.configure(text=(
                f"Mom 20d: {rd.get('momentum_20d',0):+.2f}%  "
                f"Vol: {rd.get('realized_vol_20d',0):.1f}%\n"
                f"Conf thr: {rd.get('confidence_threshold',0.75):.2f}  "
                f"Bias: {rd.get('side_bias','both')}"
            ))
        else:
            self._regime_lbl.configure(text="NO REGIME DATA", text_color=C.GHOST)
            self._regime_detail.configure(text="Run the agent to detect regime")

        # Model edge
        lr = _load_json(_LEARN_REPORT)
        if lr:
            wr = lr.get("win_rate", 0)
            ar = lr.get("avg_r") or lr.get("in_sample_acc") or 0
            ct = lr.get("suggested_confidence_threshold", 0.75)
            self.m_winrate.set(f"{wr*100:.1f}%",
                                color=C.LIFE if wr > 0.5 else C.DEATH)
            self.m_avgr.set(f"{ar:.3f}")
            self.m_conf.set(f"{ct:.2f}")
        else:
            try:
                from src.learning.trade_journal import TradeJournal
                s = TradeJournal().stats()
                wr = s.get("win_rate", 0)
                self.m_winrate.set(f"{wr*100:.1f}%",
                                    color=C.LIFE if wr > 0.5 else C.DEATH)
                self.m_avgr.set(f"{s.get('avg_r',0):.3f}")
                self.m_conf.set("0.75")
            except Exception:
                pass

        # Collector heartbeat
        self._refresh_collector_status()

    def _refresh_collector_status(self) -> None:
        try:
            from src.data.vector_store import VectorStore
            vs   = VectorStore()
            age  = vs.cache_age_minutes()
            meta = vs.load_latest_meta()
            if meta is None:
                self._coll_lbl.configure(text="BUILDING…", text_color=C.OMEN)
                self._coll_detail.configure(
                    text="First collection in progress.\nTakes ~60s for 200 symbols.")
            else:
                cnt = meta.get("count", 0)
                snaps = meta.get("snapshots", [])
                bulls  = sum(1 for s in snaps if s.get("signal") == "bullish")
                bears  = sum(1 for s in snaps if s.get("signal") == "bearish")
                alerts = sum(1 for s in snaps if abs(s.get("score", 0)) > 50)
                color = C.LIFE if age < 30 else (C.OMEN if age < 60 else C.DEATH)
                self._coll_lbl.configure(
                    text=f"{cnt} symbols  ({age:.0f}m ago)", text_color=color)
                self._coll_detail.configure(
                    text=f"↑ {bulls} bullish  ↓ {bears} bearish\n"
                         f"⚡ {alerts} high-conviction alerts")
        except Exception:
            self._coll_lbl.configure(text="——", text_color=C.GHOST)

    # ── Cache reload from disk ────────────────────────────────────────

    def _reload_cache(self) -> None:
        """Read latest.json from VectorStore and update UI."""
        try:
            from src.data.vector_store import VectorStore
            vs      = VectorStore()
            snaps   = vs.load_latest()
            age     = vs.cache_age_minutes()
            if snaps:
                self._snapshots = snaps
                self._populate_watchlist(snaps)
                self._refresh_learning_feed()
            # If cache is stale, kick off a background scan
            if age > self.STALE_MINUTES and not self._scan_running:
                threading.Thread(target=self._bg_collect, daemon=True).start()
        except Exception:
            pass
        self.after(self.CACHE_RELOAD_MS, self._reload_cache)

    def _bg_collect(self) -> None:
        """Lightweight background yfinance scan when cache is stale."""
        self._scan_running = True
        try:
            from src.data.market_collector import MarketCollector
            snaps = MarketCollector(period="2mo").run_full()
            snap_dicts = [s.to_dict() for s in snaps]
            try:
                self.after(0, lambda sd=snap_dicts: self._apply_fresh(sd))
            except RuntimeError:
                pass
        except Exception:
            pass
        finally:
            self._scan_running = False

    def _apply_fresh(self, snaps: List[Dict]) -> None:
        self._snapshots = snaps
        self._populate_watchlist(snaps)
        self._refresh_collector_status()
        self._refresh_learning_feed()

    def _force_scan(self) -> None:
        if self._scan_running:
            self.app.toast("Scan already running…")
            return
        self._force_scan_btn.configure(state="disabled", text="Scanning…")
        self.app.toast(f"{G.RUNE_R}  Market scan started (~60s)…")
        def _done():
            try:
                self._force_scan_btn.configure(state="normal", text="⟳  Scan Now")
            except RuntimeError:
                pass
        def _run():
            self._bg_collect()
            try:
                self.after(0, _done)
            except RuntimeError:
                pass
        threading.Thread(target=_run, daemon=True).start()

    # ── Signal log ────────────────────────────────────────────────────

    def _load_signals(self) -> None:
        self.sig_tree.delete(*self.sig_tree.get_children())

        # From trade journal (preferred)
        try:
            from src.learning.trade_journal import TradeJournal
            entries = list(reversed(TradeJournal()._load_all()))[:20]
            for e in entries:
                outcome = e.get("outcome") or e.get("status", "open")
                tag  = {"win": "win", "loss": "loss"}.get(outcome, "open")
                ts   = (e.get("open_ts") or e.get("signal_ts") or "")[:16]
                self.sig_tree.insert("", "end", tags=(tag,), values=(
                    ts, e.get("symbol",""), e.get("side",""),
                    f"{e.get('confidence',0):.2f}",
                    f"${e.get('entry_price',0):,.2f}",
                    f"${e.get('stop_price',0):,.2f}",
                    f"${e.get('tp_price',0):,.2f}",
                    outcome.upper(),
                ))
            return
        except Exception:
            pass

        # Fallback: raw signal JSON files
        for p in sorted(_SIGNAL_DIR.glob("*.json"))[-20:][::-1]:
            try:
                d    = json.loads(p.read_text())
                ts   = (d.get("timestamp") or p.stem)[:16]
                sym  = d.get("asset", "?")
                dir_ = d.get("direction", "?")
                conf = float(d.get("confidence", 0))
                self.sig_tree.insert("", "end", tags=("dry_run",), values=(
                    ts, sym, dir_, f"{conf:.2f}",
                    f"${float(d.get('entry_price',0)):,.2f}",
                    f"${float(d.get('stop_loss',0)):,.2f}",
                    f"${float(d.get('take_profit',0)):,.2f}",
                    "DRY-RUN",
                ))
            except Exception:
                pass

    # ── Learning feed ─────────────────────────────────────────────────

    def _refresh_learning_feed(self) -> None:
        lines: List[str] = []

        # Collector summary
        try:
            from src.data.vector_store import VectorStore
            lines.append(f"  {G.CROSS}  DATA COLLECTOR")
            lines.append(VectorStore().summary())
            lines.append("")
        except Exception:
            pass

        # Model report
        lr = _load_json(_LEARN_REPORT)
        if lr:
            lines.append(f"  {G.CROSS}  MODEL UPDATE")
            lines.append(f"  Retrained: {lr.get('retrained_at','?')[:16]}")
            lines.append(f"  Rows: {lr.get('training_rows',0)}  "
                         f"Acc: {lr.get('in_sample_acc',0)*100:.1f}%")
            lines.append(f"  Win rate: {lr.get('win_rate',0)*100:.1f}%  "
                         f"Conf: {lr.get('suggested_confidence_threshold',0.75):.2f}")
            lines.append("")

        # Journal stats
        try:
            from src.learning.trade_journal import TradeJournal
            s = TradeJournal().stats()
            lines.append(f"  {G.CROSS}  TRADE JOURNAL")
            lines.append(f"  Closed: {s['total_closed']}  Open: {s['open']}  "
                         f"Win: {s['win_rate']*100:.1f}%")
            lines.append(f"  Avg R: {s['avg_r']:.3f}  "
                         f"P&L: ${s['total_pnl_usd']:+,.2f}")
            lines.append("")
        except Exception:
            pass

        # Postmortem lessons
        try:
            from src.learning.postmortem_db import PostmortemDB
            lessons = PostmortemDB().find(min_severity=3)[:4]
            if lessons:
                lines.append(f"  {G.CROSS}  TOP LESSONS")
                for l in lessons:
                    lines.append(f"  ⚠  {l.title}")
                    lines.append(f"     → {l.mitigation[:80]}")
                lines.append("")
        except Exception:
            pass

        lines.append(f"  {G.CROSS}  COVERAGE")
        lines.append("  200+ symbols  ·  every 20 min  ·  32-dim feature vectors")
        lines.append("  Index/Sector/Commodity/Leveraged/Stock/Crypto/SmallCap")
        lines.append("  yfinance OHLCV → RSI/MACD/BB/ATR/vol/momentum/RS")

        self._learn_box.set_text("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────
    # Extended-hours toggle
    # ──────────────────────────────────────────────────────────────────

    def _on_ext_toggle(self) -> None:
        enabled = self.v_ext_hours.get()
        try:
            self.app._extended_hours = enabled
        except Exception:
            pass
        self.app.toast(f"Extended-hours {'ENABLED' if enabled else 'DISABLED'}.")
