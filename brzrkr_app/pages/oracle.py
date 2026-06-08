"""Oracle — AI trading analyst chat powered by Ollama (local, free, no API key).

Connects to Ollama running on localhost:11434. Works with any model you
have pulled — llama3, qwen3, mistral, etc. Completely private, no
internet needed, no API key required.

Install Ollama: https://ollama.com
Pull a model:   ollama pull llama3

Ask it anything: explain a trade, research a company, interpret a signal,
compare strategies, or just talk through risk.
"""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import customtkinter as ctk
import requests

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS

ROOT        = Path(__file__).resolve().parent.parent.parent
SIGNALS_DIR = ROOT / "data" / "signals"

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "llama3"   # overridden by whatever is actually installed

_SYSTEM = """You are the BRZRKR Oracle — an expert AI trading analyst embedded inside the BRZRKR autonomous trading platform.

You have deep expertise in:
• Equity and options markets, technical and fundamental analysis
• Quantitative trading strategies, risk management, and position sizing
• Reading and interpreting trading signals, P&L data, and market regimes
• Explaining complex financial concepts in plain language

The user is a trader using the BRZRKR platform. You have been given their current portfolio snapshot. Use it to give specific, grounded answers. Be direct and concise — this is a trading terminal, not a classroom. Lead with the answer, follow with reasoning.

When asked about a specific stock or company, give a structured analysis: what the company does, key metrics/catalysts, current technical picture, and a directional view. Always note relevant risks.

Frame everything as analysis, not financial advice. Be opinionated and clear."""


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _ollama_models() -> List[str]:
    """Return list of locally available model names. Empty list if offline."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _ollama_chat(model: str, messages: List[Dict], system: str) -> str:
    """Send a chat request to Ollama and return the full reply text."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_ctx": 4096},
    }
    # 300s timeout: first load can be slow while the model is read from disk
    r = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json=payload,
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Portfolio context
# ---------------------------------------------------------------------------

def _build_context(snap: Optional[Dict]) -> str:
    parts = []
    if snap:
        eq  = snap.get("equity")
        pos = snap.get("positions", [])
        ord_ = snap.get("orders", [])
        if eq:
            parts.append(f"Account equity: ${eq:,.2f}")
        if pos:
            parts.append(f"\nOpen positions ({len(pos)}):")
            for p in pos[:10]:
                sym  = p.get("symbol", "?")
                side = p.get("side", "?")
                qty  = p.get("qty", 0)
                upnl = float(p.get("unrealized_plpc", 0)) * 100
                mv   = p.get("market_value", 0)
                parts.append(
                    f"  {sym} {side} {qty} shares  MV=${float(mv):,.0f}  "
                    f"unrealised={upnl:+.2f}%"
                )
        if ord_:
            parts.append(f"\nOpen orders ({len(ord_)}):")
            for o in ord_[:5]:
                parts.append(
                    f"  {o.get('symbol')} {o.get('side')} "
                    f"{o.get('qty')} @ {o.get('type')}"
                )

    sig_files = (
        sorted(SIGNALS_DIR.glob("*.json"), reverse=True)[:5]
        if SIGNALS_DIR.exists() else []
    )
    if sig_files:
        parts.append("\nRecent signals:")
        for sf in sig_files:
            try:
                s = json.loads(sf.read_text())
                conf = float(s.get("confidence", 0))
                parts.append(
                    f"  {s.get('asset')} {s.get('direction')} "
                    f"entry=${float(s.get('entry_price', 0)):.2f} "
                    f"conf={conf:.0%} ({s.get('timestamp', '')[:10]})"
                )
            except Exception:
                pass

    return "\n".join(parts) if parts else "No live portfolio data available."


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class OraclePage(ctk.CTkFrame):
    """Local AI chat — powered by Ollama, zero API key required."""

    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._snap: Optional[Dict] = None
        self._history: List[Dict[str, str]] = []
        self._thinking = False
        self._model: str = DEFAULT_MODEL
        self._models: List[str] = []

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_body()
        self._build_input()

        # Detect available models in background so startup isn't blocked
        threading.Thread(target=self._detect_models, daemon=True).start()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 0))
        row.grid_columnconfigure(1, weight=1)

        # Left: title
        left = ctk.CTkFrame(row, fg_color="transparent")
        left.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            left,
            text=f"{G.SUN}  ORACLE",
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=24, weight="bold"),
            text_color=C.OMEN,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            left,
            text="Local AI analyst  ·  no API key needed",
            font=ctk.CTkFont(family=FONT_SANS[0], size=11),
            text_color=C.ASH,
        ).pack(side="left")

        # Right: model selector + status dot
        right = ctk.CTkFrame(row, fg_color=C.PANEL, corner_radius=8)
        right.grid(row=0, column=2, sticky="e")

        ctk.CTkLabel(
            right,
            text="Model",
            font=ctk.CTkFont(family=FONT_SANS[0], size=10),
            text_color=C.ASH,
        ).grid(row=0, column=0, padx=(12, 6), pady=8)

        self._model_var = ctk.StringVar(value="detecting…")
        self._model_menu = ctk.CTkOptionMenu(
            right,
            variable=self._model_var,
            values=["detecting…"],
            width=180,
            height=28,
            fg_color=C.OBSIDIAN,
            button_color=C.BLOOD_DIM,
            button_hover_color=C.BLOOD,
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
            command=self._on_model_change,
        )
        self._model_menu.grid(row=0, column=1, padx=(0, 6), pady=8)

        self._status_dot = ctk.CTkLabel(
            right,
            text=G.DOT_DIM,
            text_color=C.GHOST,
            font=ctk.CTkFont(size=14),
        )
        self._status_dot.grid(row=0, column=2, padx=(0, 10))

        # Separator
        ctk.CTkFrame(self, height=1, fg_color=C.BORDER).grid(
            row=1, column=0, sticky="ew", padx=24, pady=(12, 0))

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=2, column=0, sticky="nsew", padx=24, pady=12)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=250, weight=0)

        # ── Chat transcript ────────────────────────────────────────────
        chat_card = ctk.CTkFrame(body, fg_color=C.PANEL, corner_radius=10)
        chat_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        chat_card.grid_rowconfigure(0, weight=1)
        chat_card.grid_columnconfigure(0, weight=1)

        self._chat = tk.Text(
            chat_card,
            bg=C.PANEL, fg=C.BONE,
            insertbackground=C.OMEN,
            selectbackground=C.BLOOD_DIM,
            font=(FONT_MONO[0], 12),
            wrap="word",
            relief="flat",
            padx=18, pady=16,
            state="disabled",
            cursor="arrow",
        )
        self._chat.grid(row=0, column=0, sticky="nsew")

        sb = ctk.CTkScrollbar(
            chat_card, command=self._chat.yview,
            button_color=C.BLOOD_DIM,
            button_hover_color=C.BLOOD,
        )
        sb.grid(row=0, column=1, sticky="ns")
        self._chat.configure(yscrollcommand=sb.set)

        self._chat.tag_configure("you_lbl",
            foreground=C.BLOOD_HI,
            font=(FONT_SANS[0], 10, "bold"),
        )
        self._chat.tag_configure("you_txt",
            foreground=C.BONE,
            font=(FONT_MONO[0], 12),
            lmargin1=18, lmargin2=18,
        )
        self._chat.tag_configure("ai_lbl",
            foreground=C.OMEN,
            font=(FONT_SANS[0], 10, "bold"),
        )
        self._chat.tag_configure("ai_txt",
            foreground=C.PAPER,
            font=(FONT_MONO[0], 12),
            lmargin1=18, lmargin2=18,
            spacing3=4,
        )
        self._chat.tag_configure("dim",
            foreground=C.GHOST,
            font=(FONT_MONO[0], 10, "italic"),
            lmargin1=18, lmargin2=18,
        )
        self._chat.tag_configure("sep",
            foreground=C.IRON,
        )
        self._chat.tag_configure("err",
            foreground=C.EMBER,
            font=(FONT_MONO[0], 11),
            lmargin1=18, lmargin2=18,
        )

        self._show_welcome()

        # ── Context sidebar ────────────────────────────────────────────
        ctx_card = ctk.CTkFrame(body, fg_color=C.PANEL, corner_radius=10, width=250)
        ctx_card.grid(row=0, column=1, sticky="nsew")
        ctx_card.grid_propagate(False)
        ctx_card.grid_columnconfigure(0, weight=1)
        ctx_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            ctx_card,
            text=f"{G.DIAMOND}  PORTFOLIO CONTEXT",
            font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
            text_color=C.PARCHMENT,
        ).grid(row=0, column=0, padx=14, pady=(14, 4), sticky="w")

        self._ctx = tk.Text(
            ctx_card,
            bg=C.PANEL, fg=C.ASH,
            font=(FONT_MONO[0], 10),
            wrap="word",
            relief="flat",
            padx=12, pady=4,
            state="disabled",
            cursor="arrow",
        )
        self._ctx.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))

        ctk.CTkButton(
            ctx_card,
            text=f"{G.GEAR}  Refresh",
            height=28,
            fg_color=C.IRON,
            hover_color=C.PANEL_HI,
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_SANS[0], size=10),
            command=self._refresh_ctx,
        ).grid(row=2, column=0, padx=10, pady=8, sticky="ew")

        self._refresh_ctx()

    def _build_input(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=3, column=0, sticky="ew", padx=24, pady=(0, 18))
        row.grid_columnconfigure(0, weight=1)

        # Quick prompts
        qrow = ctk.CTkFrame(row, fg_color="transparent")
        qrow.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        for label in [
            "Explain my positions",
            "What's the market regime?",
            "Biggest risk right now?",
            "Explain Kelly sizing",
            "What is SPY doing?",
            "Clear",
        ]:
            ctk.CTkButton(
                qrow,
                text=label,
                height=26,
                fg_color=C.IRON,
                hover_color=C.BLOOD_DIM,
                text_color=C.PARCHMENT,
                font=ctk.CTkFont(family=FONT_SANS[0], size=10),
                corner_radius=13,
                command=lambda t=label: self._quick(t),
            ).pack(side="left", padx=(0, 6))

        # Text box
        self._input = ctk.CTkTextbox(
            row,
            height=66,
            fg_color=C.OBSIDIAN,
            border_color=C.BORDER,
            border_width=1,
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=13),
            corner_radius=8,
            wrap="word",
        )
        self._input.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self._input.bind("<Return>", self._on_enter)
        self._input.bind("<Shift-Return>", lambda e: None)

        btns = ctk.CTkFrame(row, fg_color="transparent")
        btns.grid(row=1, column=1, sticky="ns")

        self._send_btn = ctk.CTkButton(
            btns,
            text=f"{G.EXEC}  Ask",
            width=88,
            height=34,
            fg_color=C.OMEN,
            hover_color="#d4a830",
            text_color=C.VOID,
            font=ctk.CTkFont(family=FONT_SANS[0], size=13, weight="bold"),
            corner_radius=8,
            command=self._send,
        )
        self._send_btn.pack(fill="x", pady=(0, 6))

        ctk.CTkButton(
            btns,
            text="Clear",
            width=88,
            height=26,
            fg_color=C.IRON,
            hover_color=C.PANEL_HI,
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_SANS[0], size=11),
            corner_radius=8,
            command=self._clear,
        ).pack(fill="x")

    # ------------------------------------------------------------------
    # Model detection
    # ------------------------------------------------------------------

    def _detect_models(self) -> None:
        models = _ollama_models()
        self.after(0, lambda: self._on_models(models))

    def _on_models(self, models: List[str]) -> None:
        self._models = models
        if models:
            self._model = models[0]
            self._model_var.set(models[0])
            self._model_menu.configure(values=models)
            self._status_dot.configure(text=G.DOT_ON, text_color=C.LIFE)
        else:
            self._model_var.set("Ollama not running")
            self._model_menu.configure(values=["Ollama not running"])
            self._status_dot.configure(text=G.DOT_OFF, text_color=C.GHOST)
            self._append_err(
                "Ollama is not running.\n\n"
                "Open Ollama from your Applications folder or menu bar,\n"
                "then click the model dropdown above to refresh.\n\n"
                "If not installed: https://ollama.com/download/mac"
            )

    def _on_model_change(self, value: str) -> None:
        if value not in ("detecting…", "Ollama not running"):
            self._model = value

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _on_enter(self, _event) -> str:
        self._send()
        return "break"

    def _quick(self, text: str) -> None:
        if text == "Clear":
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
        if not self._models:
            self._append_err("No Ollama models available. See instructions above.")
            return

        self._input.delete("1.0", "end")
        self._append_you(text)
        self._history.append({"role": "user", "content": text})
        self._set_thinking(True)

        model   = self._model
        history = list(self._history[-20:])
        ctx     = _build_context(self._snap)
        system  = f"{_SYSTEM}\n\n--- LIVE PORTFOLIO ---\n{ctx}"

        def _worker():
            try:
                reply = _ollama_chat(model, history, system)
                self._history.append({"role": "assistant", "content": reply})
                self.after(0, lambda: self._on_reply(reply))
            except Exception as exc:
                self.after(0, lambda: self._on_err(str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_reply(self, text: str) -> None:
        self._set_thinking(False)
        self._remove_placeholder()
        self._append_ai(text)

    def _on_err(self, msg: str) -> None:
        self._set_thinking(False)
        self._remove_placeholder()
        self._append_err(f"Error: {msg}")

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _show_welcome(self) -> None:
        self._append_ai(
            "Oracle online — powered by Ollama (local AI, no internet, no API key).\n\n"
            "I have your live portfolio context. Ask me anything:\n\n"
            "  • \"Explain my current positions and their risk\"\n"
            "  • \"What is NVDA's outlook right now?\"\n"
            "  • \"Why might the agent go long on AMD?\"\n"
            "  • \"Explain the Kelly criterion in simple terms\"\n\n"
            "First message may take 30–60 seconds while Ollama loads the model.\n"
            "Make sure Ollama.app is running (it lives in your menu bar)."
        )

    def _append_you(self, text: str) -> None:
        self._chat.configure(state="normal")
        self._chat.insert("end", "\n")
        self._chat.insert("end",
            f"  YOU  {datetime.now().strftime('%H:%M')}\n", "you_lbl")
        self._chat.insert("end", f"{text}\n", "you_txt")
        self._chat.configure(state="disabled")
        self._chat.see("end")

    def _append_ai(self, text: str) -> None:
        self._chat.configure(state="normal")
        self._chat.insert("end", "\n")
        self._chat.insert("end",
            f"  {G.SUN} ORACLE  {datetime.now().strftime('%H:%M')}\n", "ai_lbl")
        self._chat.insert("end", f"{text}\n", "ai_txt")
        self._chat.insert("end", f"\n{'─'*56}\n", "sep")
        self._chat.configure(state="disabled")
        self._chat.see("end")

    def _append_err(self, text: str) -> None:
        self._chat.configure(state="normal")
        self._chat.insert("end", f"\n  {G.SKULL}  {text}\n", "err")
        self._chat.configure(state="disabled")
        self._chat.see("end")

    _PH = "__thinking__"

    def _set_thinking(self, on: bool) -> None:
        self._thinking = on
        self._send_btn.configure(
            state="disabled" if on else "normal",
            text="  …" if on else f"{G.EXEC}  Ask",
        )
        if on:
            self._chat.configure(state="normal")
            self._chat.insert("end", "\n")
            self._chat.insert("end",
                f"  {G.SUN} ORACLE\n", "ai_lbl")
            self._chat.insert("end",
                f"  Thinking… (first message may take 30–60s while the model loads)\n", "dim")
            self._chat.mark_set(self._PH, "end-2l linestart")
            self._chat.configure(state="disabled")
            self._chat.see("end")

    def _remove_placeholder(self) -> None:
        try:
            self._chat.configure(state="normal")
            start = self._chat.index(self._PH)
            self._chat.delete(start, "end")
            self._chat.configure(state="disabled")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context panel
    # ------------------------------------------------------------------

    def _refresh_ctx(self) -> None:
        text = _build_context(self._snap)
        self._ctx.configure(state="normal")
        self._ctx.delete("1.0", "end")
        self._ctx.insert("end", text)
        self._ctx.configure(state="disabled")

    def _clear(self) -> None:
        self._history.clear()
        self._chat.configure(state="normal")
        self._chat.delete("1.0", "end")
        self._chat.configure(state="disabled")
        self._append_ai("Conversation cleared. Portfolio context still active.")

    # ------------------------------------------------------------------
    # Live update hook
    # ------------------------------------------------------------------

    def update_from(self, snap: Dict[str, Any]) -> None:
        self._snap = snap
        self._refresh_ctx()
