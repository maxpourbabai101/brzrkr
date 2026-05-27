"""Live Trades — real-time position dashboard with TP/SL tracking.

Live Arena card updates every 3 seconds via a background broker poll.
Each open position shows:
  symbol · side · entry · current price · target (TP) · stop (SL)
  progress bar · on-track / at-risk / danger indicator · P&L %

Below the arena: full position table and recent order history.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, pnl_color
from brzrkr_app.widgets import BloodMetric, GothicCard, PageTitle, SectionHeader

_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_PID  = _ROOT / "agent.pid"
_AGENT_STOP = _ROOT / "AGENT_STOP"
_AGENT_LOG  = _ROOT / "agent.out"


def _agent_running() -> bool:
    if not _AGENT_PID.exists():
        return False
    try:
        pid = int(_AGENT_PID.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _agent_is_dry_run() -> bool:
    """Return True if the running agent was started with --dry-run."""
    if not _AGENT_PID.exists():
        return True
    try:
        pid = int(_AGENT_PID.read_text().strip())
        import subprocess
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            stderr=subprocess.DEVNULL,
        ).decode()
        return "--dry-run" in out
    except Exception:
        return True


def _get_conf_threshold() -> float:
    """Read the current confidence threshold from config.yaml."""
    try:
        import yaml
        cfg_path = _ROOT / "config" / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return float(cfg.get("signals", {}).get("confidence_threshold", 0.55))
    except Exception:
        return 0.55


def _best_recent_confidence() -> Optional[float]:
    """Scan the last 400 lines of agent.out for the highest confidence seen."""
    if not _AGENT_LOG.exists():
        return None
    try:
        lines = _AGENT_LOG.read_text(errors="replace").splitlines()[-400:]
        best = 0.0
        for line in lines:
            if "confidence" in line and "below threshold" in line:
                # format: "AMD: confidence 0.13 below threshold 0.75 — no trade"
                parts = line.split("confidence")
                if len(parts) > 1:
                    try:
                        val = float(parts[1].split()[0])
                        if val > best:
                            best = val
                    except Exception:
                        pass
        return best if best > 0 else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Status helpers
# ═══════════════════════════════════════════════════════════════════════════

def _status(
    entry: float, current: float,
    tp: Optional[float], sl: Optional[float], side: str,
) -> Tuple[str, str, str]:
    """Return (key, color, label) for a position."""
    if tp is None or sl is None or tp == entry or sl == entry:
        if side == "long":
            if current >= entry * 1.004:
                return "on_track", C.LIFE,  "● ON TRACK"
            if current <= entry * 0.997:
                return "at_risk",  C.DEATH, "● AT RISK"
        else:
            if current <= entry * 0.996:
                return "on_track", C.LIFE,  "● ON TRACK"
            if current >= entry * 1.003:
                return "at_risk",  C.DEATH, "● AT RISK"
        return "neutral", C.PARCHMENT, "○  WATCH"

    if side == "long":
        risk   = max(entry - sl, 1e-9)
        reward = max(tp - entry, 1e-9)
        pct_sl = (entry - current) / risk
        pct_tp = (current - entry) / reward
    else:
        risk   = max(sl - entry, 1e-9)
        reward = max(entry - tp, 1e-9)
        pct_sl = (current - entry) / risk
        pct_tp = (entry - current) / reward

    if pct_sl >= 0.75:
        return "danger",   C.WOUND,     "⚠  DANGER"
    if pct_sl >= 0.45:
        return "at_risk",  C.DEATH,     "● AT RISK"
    if pct_tp >= 0.25:
        return "on_track", C.LIFE,      "● ON TRACK"
    return "neutral", C.PARCHMENT, "○  WATCH"


def _get_targets(
    symbol: str, orders: List[Dict],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (tp_price, sl_price) for *symbol*.

    Sources tried in order:
      1. TradeJournal (stored at signal submission time)
      2. Bracket-order legs in the orders snapshot
    """
    try:
        from src.learning.trade_journal import TradeJournal
        entry = TradeJournal().get_open_entry(symbol)
        if entry:
            tp = entry.get("tp_price")
            sl = entry.get("stop_price")
            if tp and sl:
                return float(tp), float(sl)
    except Exception:
        pass

    for o in orders:
        if o.get("symbol") != symbol:
            continue
        tp = o.get("tp_price")
        sl = o.get("sl_price")
        if tp and sl:
            return float(tp), float(sl)

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# _ArenaRow — one row per open position
# ═══════════════════════════════════════════════════════════════════════════

class _ArenaRow(ctk.CTkFrame):
    _BAR_W = 200
    _BAR_H = 10

    def __init__(self, parent):
        super().__init__(
            parent, fg_color=C.PANEL,
            corner_radius=4, border_color=C.BORDER, border_width=1,
        )
        self.grid_columnconfigure(6, weight=1)
        kw_m  = dict(font=ctk.CTkFont(family=FONT_MONO[0], size=12))
        kw_mb = dict(font=ctk.CTkFont(family=FONT_MONO[0], size=13, weight="bold"))
        kw_s  = dict(font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"))

        self._sym    = ctk.CTkLabel(self, text="", width=78,  anchor="w",
                                     text_color=C.BONE, **kw_mb)
        self._side   = ctk.CTkLabel(self, text="", width=50,  anchor="center", **kw_s)
        self._entry  = ctk.CTkLabel(self, text="", width=108, anchor="w",
                                     text_color=C.ASH, **kw_m)
        self._cur    = ctk.CTkLabel(self, text="", width=118, anchor="w", **kw_mb)
        self._tp     = ctk.CTkLabel(self, text="", width=108, anchor="w",
                                     text_color=C.OMEN, **kw_m)
        self._sl     = ctk.CTkLabel(self, text="", width=108, anchor="w",
                                     text_color=C.WOUND, **kw_m)
        self._bar    = tk.Canvas(self, width=self._BAR_W, height=self._BAR_H,
                                  bg=C.OBSIDIAN, highlightthickness=0)
        self._status = ctk.CTkLabel(self, text="", width=96,  anchor="center", **kw_s)
        self._pnl    = ctk.CTkLabel(self, text="", width=72,  anchor="e", **kw_m)

        for i, w in enumerate([self._sym, self._side, self._entry, self._cur,
                                self._tp, self._sl, self._bar, self._status,
                                self._pnl]):
            pl = 12 if i == 0 else 4
            pr = 14 if i == 8 else 4
            w.grid(row=0, column=i, padx=(pl, pr), pady=8)

    def update(self, pos: Dict, tp: Optional[float], sl: Optional[float]) -> None:
        sym     = pos["symbol"]
        side    = str(pos.get("side", "long")).lower()
        entry   = float(pos["avg_entry_price"])
        current = float(pos["current_price"])
        pnl     = float(pos["unrealized_pl"])
        pnl_pc  = float(pos.get("unrealized_plpc", 0)) * 100.0

        self._sym.configure(text=sym)
        self._side.configure(text=side.upper(),
                              text_color=C.LIFE if side == "long" else C.DEATH)
        self._entry.configure(text=f"IN  ${entry:,.2f}")
        arrow   = "↑" if current >= entry else "↓"
        cur_col = C.LIFE if current >= entry else C.DEATH
        self._cur.configure(text=f"{arrow}  ${current:,.2f}", text_color=cur_col)
        self._tp.configure(text=f"◎  ${tp:,.2f}" if tp else "◎  ——")
        self._sl.configure(text=f"✕  ${sl:,.2f}" if sl else "✕  ——")
        self._draw_bar(entry, current, tp, sl, side)
        _, sc, st = _status(entry, current, tp, sl, side)
        self._status.configure(text=st, text_color=sc)
        self._pnl.configure(text=f"{pnl_pc:+.2f}%", text_color=pnl_color(pnl))

    def _draw_bar(self, entry: float, current: float,
                  tp: Optional[float], sl: Optional[float], side: str) -> None:
        c = self._bar
        c.delete("all")
        w, h = self._BAR_W, self._BAR_H
        c.create_rectangle(0, 0, w, h, fill=C.IRON, outline="")

        if tp is None or sl is None or tp == sl:
            mid   = w // 2
            swing = max(abs(entry) * 0.04, 1e-9)
            clamp = max(min((current - entry) / swing, 1.0), -1.0)
            bx    = int(mid + clamp * mid * 0.9)
            color = C.LIFE if current >= entry else C.DEATH
            lo, hi = (mid, bx) if bx >= mid else (bx, mid)
            c.create_rectangle(lo, 1, hi, h - 1, fill=color, outline="")
            c.create_rectangle(mid - 1, 0, mid + 1, h, fill=C.BONE, outline="")
            return

        if side == "long":
            total = tp - sl
            if total <= 0:
                return
            ex  = int((entry   - sl) / total * w)
            cur = int(min(max((current - sl) / total, 0.0), 1.0) * w)
        else:
            total = sl - tp
            if total <= 0:
                return
            ex  = int((sl - entry)   / total * w)
            cur = int(min(max((sl - current) / total, 0.0), 1.0) * w)

        c.create_rectangle(0, 0, max(ex - 2, 0), h, fill="#2a0606", outline="")
        c.create_rectangle(min(ex + 2, w), 0, w, h, fill="#051a05", outline="")

        fill_col = C.LIFE if cur >= ex else C.DEATH
        lo, hi = (ex, cur) if cur >= ex else (cur, ex)
        c.create_rectangle(lo, 1, hi, h - 1, fill=fill_col, outline="")
        c.create_rectangle(ex  - 1, 0, ex  + 1, h, fill=C.BONE,     outline="")
        c.create_rectangle(cur - 1, 0, cur + 1, h, fill=C.BLOOD_HI, outline="")


# ═══════════════════════════════════════════════════════════════════════════
# TradesPage
# ═══════════════════════════════════════════════════════════════════════════

_ARENA_HEADERS = [
    ("SYMBOL",    78),  ("SIDE",      50),  ("ENTRY",     108),
    ("CURRENT",   118), ("TARGET TP", 108), ("STOP SL",   108),
    ("PROGRESS",  200), ("STATUS",     96), ("P&L %",      72),
]


class TradesPage(ctk.CTkFrame):
    REFRESH_MS = 3_000

    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._executor          = None
        self._last_positions:   List[Dict] = []
        self._last_orders:      List[Dict] = []
        self._last_equity:      float      = 0.0
        self._arena_rows:       List[_ArenaRow] = []
        self._refresh_running:  bool       = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=1)

        PageTitle(self, f"{G.RUNE_F} Living Trades",
                   subtitle="arena · positions · orders",
                   ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self._build_agent_banner(row=1)   # also starts 5 s banner refresh chain
        self._build_metrics(row=2)
        self._build_arena(row=3)
        self._build_pos_table(row=4)
        self._build_ord_table(row=5)

        self.after(self.REFRESH_MS, self._live_refresh)

    # ------------------------------------------------------------------
    # Agent status banner
    # ------------------------------------------------------------------

    def _build_agent_banner(self, row: int) -> None:
        """Compact strip showing agent mode, threshold, best signal."""
        banner = ctk.CTkFrame(
            self, fg_color=C.PANEL, corner_radius=4,
            border_color=C.BORDER, border_width=1,
        )
        banner.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        banner.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(banner, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="ew", padx=14, pady=6)
        for i in range(5):
            inner.grid_columnconfigure(i, weight=1)

        kw_label = dict(
            font=ctk.CTkFont(family=FONT_SANS[0], size=9, weight="bold"),
            anchor="w",
        )
        kw_val = dict(
            font=ctk.CTkFont(family=FONT_MONO[0], size=11, weight="bold"),
            anchor="w",
        )

        # Col 0 — Agent status
        ctk.CTkLabel(inner, text="AGENT", text_color=C.SIGIL, **kw_label
                     ).grid(row=0, column=0, sticky="w")
        self._banner_agent = ctk.CTkLabel(
            inner, text="checking…", text_color=C.ASH, **kw_val)
        self._banner_agent.grid(row=1, column=0, sticky="w")

        # Col 1 — Mode
        ctk.CTkLabel(inner, text="MODE", text_color=C.SIGIL, **kw_label
                     ).grid(row=0, column=1, sticky="w")
        self._banner_mode = ctk.CTkLabel(
            inner, text="—", text_color=C.ASH, **kw_val)
        self._banner_mode.grid(row=1, column=1, sticky="w")

        # Col 2 — Confidence threshold
        ctk.CTkLabel(inner, text="CONF THRESHOLD", text_color=C.SIGIL, **kw_label
                     ).grid(row=0, column=2, sticky="w")
        self._banner_thresh = ctk.CTkLabel(
            inner, text="—", text_color=C.BONE, **kw_val)
        self._banner_thresh.grid(row=1, column=2, sticky="w")

        # Col 3 — Best recent signal
        ctk.CTkLabel(inner, text="BEST SIGNAL (last scan)", text_color=C.SIGIL, **kw_label
                     ).grid(row=0, column=3, sticky="w")
        self._banner_best = ctk.CTkLabel(
            inner, text="—", text_color=C.BONE, **kw_val)
        self._banner_best.grid(row=1, column=3, sticky="w")

        # Col 4 — hint
        self._banner_hint = ctk.CTkLabel(
            inner, text="", text_color=C.GHOST,
            font=ctk.CTkFont(family=FONT_SANS[0], size=9),
            anchor="e", wraplength=220,
        )
        self._banner_hint.grid(row=0, column=4, rowspan=2, sticky="e")

        self._refresh_banner()

    def _refresh_banner(self) -> None:
        """Update agent banner labels — runs on main thread every 5 s."""
        try:
            running = _agent_running()
            thresh  = _get_conf_threshold()
            best    = _best_recent_confidence()

            if running:
                dry = _agent_is_dry_run()
                self._banner_agent.configure(
                    text=f"● RUNNING  (PID {_AGENT_PID.read_text().strip()})",
                    text_color=C.LIFE,
                )
                if dry:
                    self._banner_mode.configure(
                        text="DRY-RUN", text_color=C.OMEN)
                    self._banner_hint.configure(
                        text="Dry-run: agent evaluates but submits NO orders.\n"
                             "Go to STATUS → uncheck Dry run → Stop & restart.",
                        text_color=C.OMEN,
                    )
                else:
                    self._banner_mode.configure(
                        text="● EXECUTE (paper)", text_color=C.LIFE)
                    self._banner_hint.configure(
                        text="Agent is live on paper account — orders submitted automatically.",
                        text_color=C.LIFE,
                    )
            else:
                self._banner_agent.configure(
                    text="○ OFFLINE", text_color=C.DEATH)
                self._banner_mode.configure(text="—", text_color=C.ASH)
                self._banner_hint.configure(
                    text="Agent not running.  Go to STATUS → Start  (or Start Auto-Trader).",
                    text_color=C.ASH,
                )

            self._banner_thresh.configure(text=f"{thresh:.2f}")

            if best is not None:
                color = C.LIFE if best >= thresh else C.DEATH
                gap_pct = int((best / thresh) * 100)
                self._banner_best.configure(
                    text=f"{best:.2f}  ({gap_pct}% of threshold)",
                    text_color=color,
                )
            else:
                self._banner_best.configure(text="no scan yet", text_color=C.ASH)

        except Exception:
            pass

        try:
            self.after(5_000, self._refresh_banner)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_metrics(self, row: int) -> None:
        strip = ctk.CTkFrame(self, fg_color="transparent")
        strip.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        for i in range(3):
            strip.grid_columnconfigure(i, weight=1)
        self.m_pnl = BloodMetric(strip, "Unrealised P&L", "$0.00")
        self.m_mv  = BloodMetric(strip, "Market Value",   "$0.00")
        self.m_n   = BloodMetric(strip, "Positions Held", "0")
        self.m_pnl.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        self.m_mv.grid( row=0, column=1, padx=6,      sticky="nsew")
        self.m_n.grid(  row=0, column=2, padx=(6, 0), sticky="nsew")

    def _build_arena(self, row: int) -> None:
        card = GothicCard(self)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        card.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        SectionHeader(head, "Live Arena — active trades",
                       glyph=G.RUNE_T).grid(row=0, column=0, sticky="w")
        self._badge = ctk.CTkLabel(
            head, text="⟳  3s",
            font=ctk.CTkFont(family=FONT_SANS[0], size=9),
            text_color=C.GHOST,
        )
        self._badge.grid(row=0, column=1, padx=(0, 14), sticky="e")

        # Column headers
        hrow = ctk.CTkFrame(card, fg_color="transparent")
        hrow.grid(row=1, column=0, padx=14, pady=(2, 4), sticky="ew")
        for col_i, (label, width) in enumerate(_ARENA_HEADERS):
            pl = 12 if col_i == 0 else 4
            pr = 14 if col_i == len(_ARENA_HEADERS) - 1 else 4
            ctk.CTkLabel(
                hrow, text=label, width=width, anchor="w",
                text_color=C.SIGIL,
                font=ctk.CTkFont(family=FONT_SANS[0], size=9, weight="bold"),
            ).pack(side="left", padx=(pl, pr))

        # Scrollable rows
        self._scroll = ctk.CTkScrollableFrame(
            card, fg_color="transparent", height=180,
        )
        self._scroll.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")
        self._scroll.grid_columnconfigure(0, weight=1)

        self._placeholder = ctk.CTkLabel(
            self._scroll,
            text=f"  {G.DOT_DIM}  No open positions — market is watching.",
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
            anchor="w",
        )
        self._placeholder.grid(row=0, column=0, pady=12, sticky="ew")

    def _build_pos_table(self, row: int) -> None:
        card = GothicCard(self)
        card.grid(row=row, column=0, sticky="nsew", pady=(0, 10))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        SectionHeader(card, "Open Positions", glyph=G.RUNE_T
                       ).grid(row=0, column=0, sticky="ew")
        self.pos_tree = _make_tree(
            card,
            ("symbol", "side", "qty", "entry", "current",
             "mkt_value", "pnl_$", "pnl_%"),
        )
        self.pos_tree.grid(row=1, column=0, padx=14,
                            pady=(0, 14), sticky="nsew")

    def _build_ord_table(self, row: int) -> None:
        card = GothicCard(self)
        card.grid(row=row, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)
        SectionHeader(card, "Recent Orders", glyph=G.EXEC
                       ).grid(row=0, column=0, sticky="ew")
        self.ord_tree = _make_tree(
            card,
            ("submitted", "symbol", "side", "qty", "filled",
             "type", "status", "tp", "sl", "id"),
        )
        self.ord_tree.grid(row=1, column=0, padx=14,
                            pady=(0, 14), sticky="nsew")

    # ------------------------------------------------------------------
    # Live 3-second refresh
    # ------------------------------------------------------------------

    def _live_refresh(self) -> None:
        self.after(self.REFRESH_MS, self._live_refresh)
        if self._executor is None or self._refresh_running:
            return
        self._refresh_running = True
        threading.Thread(target=self._fetch_thread, daemon=True).start()

    def _fetch_thread(self) -> None:
        try:
            ex        = self._executor
            positions = ex.get_open_positions()
            orders    = ex.get_orders(status="all", limit=50)
            equity    = ex.get_account_equity()
            try:
                self.after(0, lambda p=positions, o=orders, e=equity:
                            self._apply(p, o, e))
            except RuntimeError:
                pass
        except Exception:
            pass
        finally:
            self._refresh_running = False

    # ------------------------------------------------------------------
    # update_from — driven by BrokerPoller (every 8 s)
    # ------------------------------------------------------------------

    def update_from(self, snap: dict) -> None:
        if snap.get("executor"):
            self._executor = snap["executor"]
        if not snap.get("ok"):
            return
        self._apply(snap["positions"], snap["orders"],
                    snap.get("equity", 0.0))

    # ------------------------------------------------------------------
    # _apply
    # ------------------------------------------------------------------

    def _apply(
        self,
        positions: List[Dict],
        orders: List[Dict],
        equity: float,
    ) -> None:
        self._last_positions = positions
        self._last_orders    = orders
        self._last_equity    = equity

        pnl = sum(p["unrealized_pl"]  for p in positions)
        mv  = sum(p["market_value"]   for p in positions)
        eq  = equity or 1.0
        self.m_pnl.set(f"${pnl:+,.2f}",
                        sub=f"{pnl / eq * 100:+.2f}% of vault",
                        color=pnl_color(pnl))
        self.m_mv.set(f"${mv:,.2f}")
        self.m_n.set(str(len(positions)))

        self._rebuild_arena(positions, orders)
        self._rebuild_pos_table(positions)
        self._rebuild_ord_table(orders)

        self._badge.configure(text_color=C.LIFE)
        self.after(400, lambda: self._badge.configure(text_color=C.GHOST))

    # ------------------------------------------------------------------
    # Arena rebuild
    # ------------------------------------------------------------------

    def _rebuild_arena(
        self, positions: List[Dict], orders: List[Dict],
    ) -> None:
        if not positions:
            for r in self._arena_rows:
                r.grid_remove()
            self._placeholder.grid()
            return

        self._placeholder.grid_remove()

        # Grow row pool if needed
        while len(self._arena_rows) < len(positions):
            self._arena_rows.append(_ArenaRow(self._scroll))

        # Hide extras
        for j in range(len(positions), len(self._arena_rows)):
            self._arena_rows[j].grid_remove()

        # Refresh visible rows
        for i, pos in enumerate(positions):
            tp, sl = _get_targets(pos["symbol"], orders)
            row = self._arena_rows[i]
            row.update(pos, tp, sl)
            row.grid(row=i, column=0, pady=(0, 4), padx=2, sticky="ew")

    # ------------------------------------------------------------------
    # Table rebuilds
    # ------------------------------------------------------------------

    def _rebuild_pos_table(self, positions: List[Dict]) -> None:
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
                    f"{p.get('unrealized_plpc', 0)*100:+.2f}%",
                ),
            )

    def _rebuild_ord_table(self, orders: List[Dict]) -> None:
        self.ord_tree.delete(*self.ord_tree.get_children())
        for o in orders:
            is_open = o["status"] in (
                "new", "accepted", "pending_new", "partially_filled")
            tag    = "open" if is_open else "closed"
            tp_str = f"${o['tp_price']:,.2f}" if o.get("tp_price") else "——"
            sl_str = f"${o['sl_price']:,.2f}" if o.get("sl_price") else "——"
            self.ord_tree.insert(
                "", "end", tags=(tag,),
                values=(
                    (o["submitted_at"] or "")[:19],
                    o["symbol"], o["side"],
                    f"{o['qty']:g}", f"{o['filled_qty']:g}",
                    o["order_type"], o["status"],
                    tp_str, sl_str,
                    o["id"][:10] + "…",
                ),
            )


# ═══════════════════════════════════════════════════════════════════════════
# Treeview factory
# ═══════════════════════════════════════════════════════════════════════════

def _make_tree(parent: ctk.CTkFrame, cols: tuple) -> ttk.Treeview:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Brzrkr.Treeview",
        background=C.OBSIDIAN, foreground=C.BONE,
        fieldbackground=C.OBSIDIAN, borderwidth=0,
        rowheight=24, font=(FONT_MONO[0], 11),
    )
    style.configure(
        "Brzrkr.Treeview.Heading",
        background=C.PANEL_HI, foreground=C.SIGIL,
        font=(FONT_SANS[0], 9, "bold"), relief="flat",
    )
    style.map(
        "Brzrkr.Treeview",
        background=[("selected", C.BLOOD_DIM)],
        foreground=[("selected", C.BONE)],
    )
    tree = ttk.Treeview(parent, columns=cols, show="headings",
                         selectmode="browse", style="Brzrkr.Treeview")
    widths = {
        "symbol": 80,  "side": 60,  "qty": 60,    "entry": 100,
        "current": 100, "mkt_value": 110, "pnl_$": 100, "pnl_%": 80,
        "submitted": 150, "filled": 60, "type": 90, "status": 110,
        "tp": 90, "sl": 90, "id": 120,
    }
    for c in cols:
        tree.heading(c, text=c.upper().replace("_", " "))
        tree.column(c, anchor="w", width=widths.get(c, 100), stretch=True)
    tree.tag_configure("gain",   foreground=C.LIFE)
    tree.tag_configure("loss",   foreground=C.DEATH)
    tree.tag_configure("open",   foreground=C.OMEN)
    tree.tag_configure("closed", foreground=C.ASH)
    return tree
