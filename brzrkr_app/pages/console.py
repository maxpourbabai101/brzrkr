"""Trade Console — manual orders, position closes, order cancels."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, pnl_color
from brzrkr_app.widgets import (
    GhostButton, GothicCard, PageTitle, RuneButton, SectionHeader,
)


class ConsolePage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._executor = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)

        PageTitle(self, f"{G.EXEC} Trade Console",
                   subtitle="raise · seal · banish").grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        # Left: new order form
        form = GothicCard(self)
        form.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
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

        self.v_symbol = field(1, "Symbol", "SPY")

        # Side
        ctk.CTkLabel(form, text="SIDE", text_color=C.PARCHMENT,
                      font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
                      anchor="w").grid(row=2, column=0, padx=(16, 8), pady=4, sticky="w")
        self.v_side = ctk.StringVar(value="long")
        side_f = ctk.CTkFrame(form, fg_color="transparent")
        side_f.grid(row=2, column=1, padx=(0, 16), pady=4, sticky="ew")
        ctk.CTkRadioButton(side_f, text="Long", variable=self.v_side,
                            value="long", fg_color=C.LIFE,
                            text_color=C.BONE, border_color=C.BORDER
                            ).pack(side="left", padx=(0, 14))
        ctk.CTkRadioButton(side_f, text="Short", variable=self.v_side,
                            value="short", fg_color=C.DEATH,
                            text_color=C.BONE, border_color=C.BORDER
                            ).pack(side="left")

        self.v_notional = field(3, "Notional ($)", "1000")
        self.v_stop = field(4, "Stop %", "1.0")
        self.v_tp = field(5, "Take Profit %", "2.0")

        # ── Session selector
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

        # Extended-hours note (shown when non-regular selected)
        self._ext_note = ctk.CTkLabel(
            form,
            text="  ⚠  Extended hours: limit order only, no bracket",
            text_color=C.OMEN,
            font=ctk.CTkFont(family=FONT_SANS[0], size=9),
            anchor="w",
        )
        # Hidden by default; shown when pre/after selected

        self.v_confirm = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(form, text="I bear the consequence",
                         variable=self.v_confirm,
                         fg_color=C.BLOOD, hover_color=C.BLOOD_HI,
                         border_color=C.BORDER, text_color=C.PARCHMENT
                         ).grid(row=8, column=0, columnspan=2,
                                padx=16, pady=(10, 4), sticky="w")

        RuneButton(form, "Strike", glyph=G.EXEC,
                    command=self._submit
                    ).grid(row=9, column=0, columnspan=2,
                            padx=16, pady=(8, 14), sticky="ew")

        # Right: position close + order cancel rows
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        pos = GothicCard(right)
        pos.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        pos.grid_columnconfigure(0, weight=1)
        pos.grid_rowconfigure(1, weight=1)
        SectionHeader(pos, "Seal Positions", glyph=G.SHIELD).grid(
            row=0, column=0, sticky="ew")
        self.pos_scroll = ctk.CTkScrollableFrame(pos, fg_color="transparent")
        self.pos_scroll.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.pos_scroll.grid_columnconfigure(0, weight=1)

        ord_card = GothicCard(right)
        ord_card.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        ord_card.grid_columnconfigure(0, weight=1)
        ord_card.grid_rowconfigure(1, weight=1)
        hdr_f = ctk.CTkFrame(ord_card, fg_color="transparent")
        hdr_f.grid(row=0, column=0, sticky="ew")
        hdr_f.grid_columnconfigure(0, weight=1)
        SectionHeader(hdr_f, "Banish Orders", glyph=G.DAGGER).grid(
            row=0, column=0, sticky="ew")
        GhostButton(hdr_f, "Banish ALL", glyph=G.SKULL,
                     command=self._cancel_all
                     ).grid(row=0, column=1, padx=(0, 16), pady=(12, 4))
        self.ord_scroll = ctk.CTkScrollableFrame(ord_card, fg_color="transparent")
        self.ord_scroll.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")
        self.ord_scroll.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(1, weight=1)

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
        if not snap.get("ok"):
            return
        self._executor = snap["executor"]

        for c in self.pos_scroll.winfo_children():
            c.destroy()
        if not snap["positions"]:
            ctk.CTkLabel(self.pos_scroll, text=f"{G.DOT_DIM}  no positions held",
                          text_color=C.ASH).grid(row=0, column=0, padx=4, pady=8, sticky="w")
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
                          font=ctk.CTkFont(family=FONT_MONO[0], size=12)
                          ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
            GhostButton(row, "Seal", width=70,
                         command=lambda s=p["symbol"]: self._close(s)
                         ).grid(row=0, column=1, padx=8, pady=4)

        for c in self.ord_scroll.winfo_children():
            c.destroy()
        open_orders = [o for o in snap["orders"]
                        if o["status"] in ("new", "accepted",
                                            "pending_new", "partially_filled")]
        if not open_orders:
            ctk.CTkLabel(self.ord_scroll, text=f"{G.DOT_DIM}  no orders pending",
                          text_color=C.ASH).grid(row=0, column=0, padx=4, pady=8, sticky="w")
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
                          font=ctk.CTkFont(family=FONT_MONO[0], size=12)
                          ).grid(row=0, column=0, sticky="ew", padx=8, pady=6)
            GhostButton(row, "Banish", width=80,
                         command=lambda oid=o["id"]: self._cancel(oid)
                         ).grid(row=0, column=1, padx=8, pady=4)

    # ------------------------------------------------------------------
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
            tp_pct = float(self.v_tp.get()) / 100.0
        except ValueError:
            self.app.toast("Numeric fields must hold numbers.")
            return

        import requests
        try:
            r = requests.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                headers={
                    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
                    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
                }, timeout=10)
            r.raise_for_status()
            entry = float((r.json().get("trade") or {}).get("p", 0))
        except Exception as exc:  # noqa: BLE001
            self.app.toast(f"Quote unreachable: {exc}")
            return
        if entry <= 0:
            self.app.toast("No price; symbol misspoken?")
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
            "risk_flags": {"manual_brzrkr": True},
        }
        try:
            from src.execution.broker import AlpacaExecutor
            ex = self._executor or AlpacaExecutor(live_money=False)
            session = self.v_session.get()
            ext_hours = session in ("pre_market", "after_hours")
            result = ex.submit_signal(sig,
                                       extended_hours=ext_hours,
                                       session=session)
            if result.submitted:
                sess_tag = f" [{session}]" if ext_hours else ""
                self.app.toast(
                    f"{G.EXEC}  Struck{sess_tag} — id {result.order_id[:10]}…")
                self.v_confirm.set(False)
                self.app.poller.trigger()
            else:
                self.app.toast(f"{G.SKULL}  Rejected: {result.reason}")
        except Exception as exc:  # noqa: BLE001
            self.app.toast(f"Strike failed: {exc}")

    def _on_session_change(self) -> None:
        sess = self.v_session.get()
        if sess in ("pre_market", "after_hours"):
            self._ext_note.grid(row=7, column=0, columnspan=2,
                                 padx=16, pady=(0, 4), sticky="w")
        else:
            self._ext_note.grid_remove()

    def _close(self, symbol: str) -> None:
        if not self._executor: return
        ok = self._executor.close_position(symbol)
        self.app.toast(f"{'Sealed' if ok else 'SEAL FAILED:'} {symbol}")
        self.app.poller.trigger()

    def _cancel(self, order_id: str) -> None:
        if not self._executor: return
        ok = self._executor.cancel_order(order_id)
        self.app.toast("Banished." if ok else "Banish failed.")
        self.app.poller.trigger()

    def _cancel_all(self) -> None:
        if not self._executor: return
        n = self._executor.cancel_all_orders()
        self.app.toast(f"Banished {n} order(s).")
        self.app.poller.trigger()
