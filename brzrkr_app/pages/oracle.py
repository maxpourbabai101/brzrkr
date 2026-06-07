"""Oracle — AI trading analyst chat powered by Claude.

Gives the user a direct conversation with an AI that has full context
about their open positions, recent trades, signals, and market data.
Ask it anything: explain a trade, research a company, interpret a
signal, or get a second opinion on risk.

The Anthropic API key can be entered directly in the page and is saved
to the project .env file so it persists across restarts.
"""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS

ROOT      = Path(__file__).resolve().parent.parent.parent
ENV_FILE  = ROOT / ".env"
SIGNALS_DIR = ROOT / "data" / "signals"

# ── System prompt given to Claude ─────────────────────────────────────────
_SYSTEM = """You are the BRZRKR Oracle — an expert AI trading analyst embedded inside the BRZRKR autonomous trading platform. You have deep expertise in:

• Equity and options markets, technical and fundamental analysis
• Quantitative trading strategies, risk management, and position sizing
• Reading and interpreting trading signals, P&L data, and market regimes
• Explaining complex financial concepts in plain language

The user is a trader using the BRZRKR platform. You have been given their current portfolio snapshot below. Use it to give specific, grounded answers. Be direct and concise — this is a trading terminal, not a classroom. Lead with the answer, follow with reasoning.

When asked about a specific stock or company, give a structured analysis covering: what the company does, key metrics/catalysts, current technical picture, and a directional view. Always note relevant risks.

Never give financial advice in the legal sense — frame everything as analysis, not instruction. But be opinionated and clear."""


def _load_env_key() -> str:
    """Read ANTHROPIC_API_KEY from the .env file."""
    if not ENV_FILE.exists():
        return ""
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _save_env_key(key: str) -> None:
    """Upsert ANTHROPIC_API_KEY in the .env file."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            new_lines.append(f"ANTHROPIC_API_KEY={key}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"ANTHROPIC_API_KEY={key}")
    ENV_FILE.write_text("\n".join(new_lines) + "\n")


def _build_portfolio_context(snap: Optional[Dict] = None) -> str:
    """Serialize the broker snapshot + recent signals into readable context."""
    parts = []

    if snap:
        eq = snap.get("equity")
        pos = snap.get("positions", [])
        orders = snap.get("orders", [])
        if eq:
            parts.append(f"Account equity: ${eq:,.2f}")
        if pos:
            parts.append(f"\nOpen positions ({len(pos)}):")
            for p in pos[:10]:
                sym  = p.get("symbol", "?")
                side = p.get("side", "?")
                qty  = p.get("qty", 0)
                upnl = p.get("unrealized_plpc", 0)
                mv   = p.get("market_value", 0)
                parts.append(
                    f"  {sym} {side} {qty} shares  MV=${mv:,.0f}  unrealised={float(upnl)*100:+.2f}%"
                )
        if orders:
            parts.append(f"\nOpen orders ({len(orders)}):")
            for o in orders[:5]:
                parts.append(
                    f"  {o.get('symbol')} {o.get('side')} {o.get('qty')} @ {o.get('type')}"
                )

    # Recent signals
    sig_files = sorted(SIGNALS_DIR.glob("*.json"), reverse=True)[:5] if SIGNALS_DIR.exists() else []
    if sig_files:
        parts.append("\nRecent signals generated:")
        for sf in sig_files:
            try:
                s = json.loads(sf.read_text())
                conf = s.get("confidence", 0)
                parts.append(
                    f"  {s.get('asset')} {s.get('direction')} "
                    f"entry=${s.get('entry_price', 0):.2f} "
                    f"conf={conf:.0%} "
                    f"({s.get('timestamp', '')[:10]})"
                )
            except Exception:
                pass

    if not parts:
        return "No live portfolio data available at this time."
    return "\n".join(parts)


class OraclePage(ctk.CTkFrame):
    """Chat interface for the BRZRKR AI trading analyst."""

    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._snap: Optional[Dict] = None
        self._history: List[Dict[str, str]] = []   # [{role, content}, ...]
        self._api_key: str = _load_env_key() or os.getenv("ANTHROPIC_API_KEY", "")
        self._thinking = False

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # ── Title row ─────────────────────────────────────────────────
        title_row = ctk.CTkFrame(self, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 0))
        title_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            title_row,
            text=f"{G.CROSS}  ORACLE  {G.CROSS}",
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=22, weight="bold"),
            text_color=C.BLOOD_HI,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_row,
            text="AI trading analyst  ·  ask anything",
            font=ctk.CTkFont(family=FONT_SANS[0], size=11),
            text_color=C.ASH,
        ).grid(row=1, column=0, sticky="w")

        # API key field (right side of title)
        key_frame = ctk.CTkFrame(title_row, fg_color=C.PANEL, corner_radius=8)
        key_frame.grid(row=0, column=2, rowspan=2, sticky="e", padx=(0, 0))

        ctk.CTkLabel(
            key_frame,
            text="Anthropic key",
            font=ctk.CTkFont(family=FONT_SANS[0], size=10),
            text_color=C.ASH,
        ).grid(row=0, column=0, padx=(10, 4), pady=(6, 2), sticky="w")

        self._key_entry = ctk.CTkEntry(
            key_frame,
            placeholder_text="sk-ant-...",
            show="•",
            width=220,
            height=28,
            fg_color=C.OBSIDIAN,
            border_color=C.BORDER,
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
        )
        if self._api_key:
            self._key_entry.insert(0, self._api_key)
        self._key_entry.grid(row=0, column=1, padx=(0, 4), pady=(6, 2))

        ctk.CTkButton(
            key_frame,
            text="Save",
            width=52,
            height=28,
            fg_color=C.BLOOD_DIM,
            hover_color=C.BLOOD,
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_SANS[0], size=11, weight="bold"),
            command=self._save_key,
        ).grid(row=0, column=2, padx=(0, 8), pady=(6, 2))

        # ── Thin separator ─────────────────────────────────────────────
        ctk.CTkFrame(self, height=1, fg_color=C.BORDER).grid(
            row=1, column=0, sticky="ew", padx=24, pady=(12, 0))

        # ── Main body: chat area (left) + context panel (right) ────────
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", padx=24, pady=12)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=240, weight=0)
        self.grid_rowconfigure(2, weight=1)

        # Chat transcript
        self._build_chat_area(body)

        # Context sidebar
        self._build_context_panel(body)

        # ── Input row ─────────────────────────────────────────────────
        self._build_input_row()

    def _build_chat_area(self, parent) -> None:
        chat_frame = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=10)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)

        self._chat_text = tk.Text(
            chat_frame,
            bg=C.PANEL, fg=C.BONE,
            insertbackground=C.BLOOD_HI,
            selectbackground=C.BLOOD_DIM,
            font=(FONT_MONO[0], 12),
            wrap="word",
            relief="flat",
            padx=16, pady=14,
            state="disabled",
            cursor="arrow",
        )
        self._chat_text.grid(row=0, column=0, sticky="nsew")

        sb = ctk.CTkScrollbar(chat_frame, command=self._chat_text.yview,
                               button_color=C.BLOOD_DIM,
                               button_hover_color=C.BLOOD)
        sb.grid(row=0, column=1, sticky="ns")
        self._chat_text.configure(yscrollcommand=sb.set)

        # Text tags for styling
        self._chat_text.tag_configure("user_label",
            foreground=C.BLOOD_HI,
            font=(FONT_SANS[0], 10, "bold"),
        )
        self._chat_text.tag_configure("user_text",
            foreground=C.BONE,
            font=(FONT_MONO[0], 12),
            lmargin1=18, lmargin2=18,
        )
        self._chat_text.tag_configure("oracle_label",
            foreground=C.OMEN,
            font=(FONT_SANS[0], 10, "bold"),
        )
        self._chat_text.tag_configure("oracle_text",
            foreground=C.PAPER,
            font=(FONT_MONO[0], 12),
            lmargin1=18, lmargin2=18,
        )
        self._chat_text.tag_configure("thinking_text",
            foreground=C.ASH,
            font=(FONT_MONO[0], 11, "italic"),
            lmargin1=18, lmargin2=18,
        )
        self._chat_text.tag_configure("separator",
            foreground=C.IRON,
        )
        self._chat_text.tag_configure("error_text",
            foreground=C.EMBER,
            font=(FONT_MONO[0], 11),
            lmargin1=18, lmargin2=18,
        )

        # Welcome message
        self._append_oracle(
            "Oracle online. I have access to your live portfolio, "
            "recent signals, and trade history.\n\n"
            "Ask me to explain a trade, research a company, interpret "
            "a signal, or anything else trading-related. Try:\n\n"
            "  • \"Explain my current positions and their risk\"\n"
            "  • \"What is NVDA's outlook right now?\"\n"
            "  • \"Why did the agent generate a long signal on AMD?\"\n"
            "  • \"How does Kelly sizing work?\""
        )

    def _build_context_panel(self, parent) -> None:
        panel = ctk.CTkFrame(parent, fg_color=C.PANEL, corner_radius=10, width=240)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text=f"{G.ORNATE}  CONTEXT",
            font=ctk.CTkFont(family=FONT_SANS[0], size=11, weight="bold"),
            text_color=C.PARCHMENT,
        ).grid(row=0, column=0, padx=14, pady=(14, 4), sticky="w")

        ctk.CTkFrame(panel, height=1, fg_color=C.BORDER).grid(
            row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        self._ctx_text = tk.Text(
            panel,
            bg=C.PANEL, fg=C.PARCHMENT,
            font=(FONT_MONO[0], 10),
            wrap="word",
            relief="flat",
            padx=10, pady=4,
            state="disabled",
            cursor="arrow",
        )
        self._ctx_text.grid(row=2, column=0, sticky="nsew", padx=4)
        panel.grid_rowconfigure(2, weight=1)

        ctk.CTkButton(
            panel,
            text=f"{G.GEAR}  Refresh context",
            height=30,
            fg_color=C.IRON,
            hover_color=C.PANEL_HI,
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_SANS[0], size=11),
            command=self._refresh_context,
        ).grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self._ctx_text.tag_configure("key",   foreground=C.OMEN)
        self._ctx_text.tag_configure("value", foreground=C.BONE)

        self._refresh_context()

    def _build_input_row(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 18))
        row.grid_columnconfigure(0, weight=1)

        # Suggested prompts
        prompts_frame = ctk.CTkFrame(row, fg_color="transparent")
        prompts_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        quick = [
            "Explain my positions",
            "Market regime?",
            "Biggest risk right now?",
            "Explain Kelly sizing",
            "Clear conversation",
        ]
        for q in quick:
            ctk.CTkButton(
                prompts_frame,
                text=q,
                height=24,
                fg_color=C.IRON,
                hover_color=C.BLOOD_DIM,
                text_color=C.PARCHMENT,
                font=ctk.CTkFont(family=FONT_SANS[0], size=10),
                corner_radius=12,
                command=lambda t=q: self._quick_prompt(t),
            ).pack(side="left", padx=(0, 6))

        # Text input
        self._input = ctk.CTkTextbox(
            row,
            height=68,
            fg_color=C.OBSIDIAN,
            border_color=C.BORDER,
            border_width=1,
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=13),
            corner_radius=8,
            wrap="word",
        )
        self._input.grid(row=1, column=0, sticky="ew", padx=(0, 10))

        btn_col = ctk.CTkFrame(row, fg_color="transparent")
        btn_col.grid(row=1, column=1, sticky="ns")

        self._send_btn = ctk.CTkButton(
            btn_col,
            text=f"{G.EXEC}  Send",
            width=90,
            height=34,
            fg_color=C.BLOOD,
            hover_color=C.BLOOD_HI,
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_SANS[0], size=13, weight="bold"),
            corner_radius=8,
            command=self._send,
        )
        self._send_btn.pack(fill="x", pady=(0, 6))

        ctk.CTkButton(
            btn_col,
            text="Clear",
            width=90,
            height=28,
            fg_color=C.IRON,
            hover_color=C.PANEL_HI,
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_SANS[0], size=11),
            corner_radius=8,
            command=self._clear,
        ).pack(fill="x")

        # Bind Enter to send, Shift+Enter for newline
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)

    # ------------------------------------------------------------------
    # Chat logic
    # ------------------------------------------------------------------

    def _on_enter(self, event) -> str:
        self._send()
        return "break"

    def _quick_prompt(self, text: str) -> None:
        if text == "Clear conversation":
            self._clear()
            return
        self._input.delete("1.0", "end")
        self._input.insert("1.0", text)
        self._send()

    def _send(self) -> None:
        if self._thinking:
            return
        text = self._input.get("1.0", "end").strip()
        if not text:
            return

        key = self._key_entry.get().strip()
        if not key:
            self._append_error(
                "No Anthropic API key set. Enter your key above and click Save."
            )
            return

        self._api_key = key
        self._input.delete("1.0", "end")
        self._append_user(text)
        self._history.append({"role": "user", "content": text})
        self._set_thinking(True)
        threading.Thread(target=self._call_api, daemon=True).start()

    def _call_api(self) -> None:
        """Run in a background thread — never touch Tk widgets here."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self._api_key)

            ctx = _build_portfolio_context(self._snap)
            system = f"{_SYSTEM}\n\n--- LIVE PORTFOLIO SNAPSHOT ---\n{ctx}"

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system,
                messages=self._history[-20:],   # keep last 20 turns for context
            )
            reply = response.content[0].text
            self._history.append({"role": "assistant", "content": reply})
            self.after(0, lambda: self._on_reply(reply))
        except Exception as exc:
            err = str(exc)
            self.after(0, lambda: self._on_error(err))

    def _on_reply(self, text: str) -> None:
        self._set_thinking(False)
        self._remove_thinking_placeholder()
        self._append_oracle(text)

    def _on_error(self, msg: str) -> None:
        self._set_thinking(False)
        self._remove_thinking_placeholder()
        self._append_error(f"API error: {msg}")

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _append_user(self, text: str) -> None:
        self._chat_text.configure(state="normal")
        self._chat_text.insert("end", "\n")
        self._chat_text.insert("end", f"  YOU  {datetime.now().strftime('%H:%M')}\n", "user_label")
        self._chat_text.insert("end", f"{text}\n", "user_text")
        self._chat_text.configure(state="disabled")
        self._chat_text.see("end")

    def _append_oracle(self, text: str) -> None:
        self._chat_text.configure(state="normal")
        self._chat_text.insert("end", "\n")
        self._chat_text.insert("end", f"  {G.CROSS} ORACLE  {datetime.now().strftime('%H:%M')}\n", "oracle_label")
        self._chat_text.insert("end", f"{text}\n", "oracle_text")
        self._chat_text.insert("end", f"\n{'─'*60}\n", "separator")
        self._chat_text.configure(state="disabled")
        self._chat_text.see("end")

    def _append_error(self, text: str) -> None:
        self._chat_text.configure(state="normal")
        self._chat_text.insert("end", f"\n  {G.SKULL}  {text}\n", "error_text")
        self._chat_text.configure(state="disabled")
        self._chat_text.see("end")

    _THINKING_MARK = "__thinking__"

    def _set_thinking(self, on: bool) -> None:
        self._thinking = on
        self._send_btn.configure(
            state="disabled" if on else "normal",
            text=f"  ..." if on else f"{G.EXEC}  Send",
        )
        if on:
            self._chat_text.configure(state="normal")
            self._chat_text.insert("end", f"\n  {G.CROSS} ORACLE\n", "oracle_label")
            self._chat_text.insert("end", "  Consulting the forge…\n",
                                   "thinking_text")
            self._chat_text.mark_set(self._THINKING_MARK, "end-2l")
            self._chat_text.configure(state="disabled")
            self._chat_text.see("end")

    def _remove_thinking_placeholder(self) -> None:
        try:
            self._chat_text.configure(state="normal")
            start = self._chat_text.index(self._THINKING_MARK)
            self._chat_text.delete(start, "end")
            self._chat_text.configure(state="disabled")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context panel
    # ------------------------------------------------------------------

    def _refresh_context(self) -> None:
        ctx = _build_portfolio_context(self._snap)
        self._ctx_text.configure(state="normal")
        self._ctx_text.delete("1.0", "end")
        self._ctx_text.insert("end", ctx)
        self._ctx_text.configure(state="disabled")

    def _save_key(self) -> None:
        key = self._key_entry.get().strip()
        if key:
            _save_env_key(key)
            self._api_key = key
            os.environ["ANTHROPIC_API_KEY"] = key
            self.app.toast("Anthropic key saved to .env")

    def _clear(self) -> None:
        self._history.clear()
        self._chat_text.configure(state="normal")
        self._chat_text.delete("1.0", "end")
        self._chat_text.configure(state="disabled")
        self._append_oracle(
            "Conversation cleared. I still have your live portfolio context.\n"
            "What would you like to know?"
        )

    # ------------------------------------------------------------------
    # Live update hook
    # ------------------------------------------------------------------

    def update_from(self, snap: Dict[str, Any]) -> None:
        self._snap = snap
        self._refresh_context()
