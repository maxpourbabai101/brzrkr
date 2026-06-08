"""Oracle — AI trading analyst chat powered by Ollama (local, free, no API key).

Connects to Ollama running on localhost:11434. Works with any model you
have pulled — llama3, qwen3, mistral, etc. Completely private, no
internet needed, no API key required.

Install Ollama: https://ollama.com
Pull a model:   ollama pull llama3

Ask it anything: explain a trade, research a company, interpret a signal,
compare strategies, or just talk through risk.

Responses stream token-by-token so you see the AI thinking in real time.
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk
import requests

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS

ROOT        = Path(__file__).resolve().parent.parent.parent
SIGNALS_DIR = ROOT / "data" / "signals"

OLLAMA_BASE   = "http://localhost:11434"
DEFAULT_MODEL = "llama3"

_SYSTEM = """You are the BRZRKR Oracle — an expert AI trading analyst embedded inside the BRZRKR autonomous trading platform.

You have deep expertise in:
• Equity and options markets, technical and fundamental analysis
• Quantitative trading strategies, risk management, and position sizing
• Reading and interpreting trading signals, P&L data, and market regimes
• Explaining complex financial concepts clearly and directly

The user is a trader running BRZRKR. You have their live portfolio snapshot below. Use it to give specific, grounded answers. Be direct — this is a trading terminal, not a classroom. Lead with the answer, follow with reasoning.

When asked about a specific stock or company: what it does, key metrics/catalysts, technical picture, directional view, key risks.

Frame everything as analysis, not financial advice. Be opinionated and clear. Keep responses concise unless depth is asked for."""


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


def _ollama_stream(
    model: str,
    messages: List[Dict],
    system: str,
    on_token: Callable[[str], None],
    on_done: Callable[[str], None],
    on_err: Callable[[str], None],
) -> None:
    """Stream a chat request from Ollama, calling on_token for each chunk.

    Runs synchronously — call from a background thread.
    on_token(text)  — called for each streamed token
    on_done(full)   — called once when the stream finishes
    on_err(msg)     — called if anything goes wrong
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": True,
        "options": {"temperature": 0.7, "num_ctx": 4096},
    }
    full_text = ""
    try:
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    on_token(token)
                if data.get("done"):
                    on_done(full_text)
                    return
        on_done(full_text)
    except Exception as exc:
        on_err(str(exc))


# ---------------------------------------------------------------------------
# Portfolio context builder
# ---------------------------------------------------------------------------

def _build_context(snap: Optional[Dict]) -> str:
    parts = []
    if snap:
        eq   = snap.get("equity")
        pos  = snap.get("positions", [])
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
        open_orders = [o for o in ord_
                       if o.get("status") in ("new", "accepted", "pending_new", "partially_filled")]
        if open_orders:
            parts.append(f"\nPending orders ({len(open_orders)}):")
            for o in open_orders[:5]:
                parts.append(
                    f"  {o.get('symbol')} {o.get('side')} "
                    f"{o.get('qty')} @ {o.get('type')}"
                )

    sig_files = (
        sorted(SIGNALS_DIR.glob("*.json"), reverse=True)[:5]
        if SIGNALS_DIR.exists() else []
    )
    if sig_files:
        parts.append("\nRecent agent signals:")
        for sf in sig_files:
            try:
                s = json.loads(sf.read_text())
                conf = float(s.get("confidence", 0))
                dd   = s.get("drawdown_prob")
                dd_s = f"  dd_prob={dd:.0%}" if dd is not None else ""
                parts.append(
                    f"  {s.get('asset')} {s.get('direction')} "
                    f"@ ${float(s.get('entry_price', 0)):.2f} "
                    f"conf={conf:.0%}{dd_s} ({s.get('timestamp', '')[:10]})"
                )
            except Exception:
                pass

    return "\n".join(parts) if parts else "No live portfolio data available."


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class OraclePage(ctk.CTkFrame):
    """Local AI chat — powered by Ollama, streaming, zero API key required."""

    def __init__(self, parent, app) -> None:
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._snap: Optional[Dict] = None
        self._history: List[Dict[str, str]] = []
        self._streaming  = False
        self._model: str = DEFAULT_MODEL
        self._models: List[str] = []

        # Mark used to anchor streaming text insertion point
        self._STREAM_MARK = "__stream_start__"

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_body()
        self._build_input()

        threading.Thread(target=self._detect_models, daemon=True).start()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_header(self) -> None:
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 0))
        row.grid_columnconfigure(1, weight=1)

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
            text="Local AI analyst  ·  no API key  ·  streams live",
            font=ctk.CTkFont(family=FONT_SANS[0], size=11),
            text_color=C.ASH,
        ).pack(side="left")

        # Model selector + status dot (clickable retry) + ↺ label
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

        # Status dot — click to re-detect Ollama
        self._status_dot = ctk.CTkButton(
            right,
            text=G.DOT_DIM,
            text_color=C.GHOST,
            fg_color="transparent",
            hover_color=C.PANEL_HI,
            width=28, height=28,
            font=ctk.CTkFont(size=14),
            command=self._retry_detect,
        )
        self._status_dot.grid(row=0, column=2, padx=(0, 2))

        ctk.CTkLabel(
            right,
            text="↺",
            text_color=C.GHOST,
            font=ctk.CTkFont(family=FONT_SANS[0], size=12),
        ).grid(row=0, column=3, padx=(0, 12))

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
        self._chat.tag_configure("stream",
            foreground=C.PAPER,
            font=(FONT_MONO[0], 12),
            lmargin1=18, lmargin2=18,
        )
        self._chat.tag_configure("cursor",
            foreground=C.OMEN,
            font=(FONT_MONO[0], 12, "bold"),
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

        # Quick prompts — context-aware, use live portfolio data
        qrow = ctk.CTkFrame(row, fg_color="transparent")
        qrow.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        for label in [
            "Analyze my open positions",
            "What's my biggest risk right now?",
            "Explain the latest signal",
            "Market regime today?",
            "Explain Kelly sizing",
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

    def _retry_detect(self) -> None:
        """Re-probe Ollama — triggered by clicking the status dot."""
        self._status_dot.configure(text="…", text_color=C.ASH)
        self._model_var.set("detecting…")
        threading.Thread(target=self._detect_models, daemon=True).start()

    def _on_models(self, models: List[str]) -> None:
        self._models = models
        if models:
            self._model = models[0]
            self._model_var.set(models[0])
            self._model_menu.configure(values=models)
            self._status_dot.configure(text=G.DOT_ON, text_color=C.LIFE)
        else:
            self._model_var.set("Ollama offline")
            self._model_menu.configure(values=["Ollama offline"])
            self._status_dot.configure(text=G.DOT_OFF, text_color=C.GHOST)
            self._append_err(
                "Ollama is not running.\n\n"
                "  1. Open Ollama.app from your Applications folder\n"
                "     (sits in your menu bar once running)\n"
                "  2. Click the  ↺  dot above to retry\n\n"
                "Not installed yet? https://ollama.com/download/mac\n"
                "Then pull a model:  ollama pull llama3"
            )

    def _on_model_change(self, value: str) -> None:
        if value not in ("detecting…", "Ollama offline"):
            self._model = value

    # ------------------------------------------------------------------
    # Chat — streaming
    # ------------------------------------------------------------------

    def _on_enter(self, _event) -> str:
        self._send()
        return "break"

    def _quick(self, text: str) -> None:
        self._input.delete("1.0", "end")
        self._input.insert("1.0", text)
        self._send()

    def _send(self) -> None:
        if self._streaming:
            return
        text = self._input.get("1.0", "end").strip()
        if not text:
            return
        if not self._models:
            self._append_err("No Ollama models found. Click the ↺ dot above to retry.")
            return

        self._input.delete("1.0", "end")
        self._append_you(text)
        self._history.append({"role": "user", "content": text})
        self._begin_stream()

        model   = self._model
        history = list(self._history[-20:])
        ctx     = _build_context(self._snap)
        system  = f"{_SYSTEM}\n\n--- LIVE PORTFOLIO ---\n{ctx}"

        def _worker():
            _ollama_stream(
                model, history, system,
                on_token=lambda t: self.after(0, lambda tok=t: self._on_token(tok)),
                on_done =lambda f: self.after(0, lambda full=f: self._on_done(full)),
                on_err  =lambda e: self.after(0, lambda msg=e: self._on_err(msg)),
            )

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Streaming state machine
    # ------------------------------------------------------------------

    def _begin_stream(self) -> None:
        """Insert Oracle label, set insertion mark, start blinking cursor."""
        self._streaming = True
        self._send_btn.configure(state="disabled", text="  ▌")

        self._chat.configure(state="normal")
        self._chat.insert("end", "\n")
        self._chat.insert("end",
            f"  {G.SUN} ORACLE  {datetime.now().strftime('%H:%M')}\n", "ai_lbl")
        # Gravity "left" keeps the mark from drifting right as we insert after it
        self._chat.mark_set(self._STREAM_MARK, "end")
        self._chat.mark_gravity(self._STREAM_MARK, "left")
        # Blinking cursor placeholder
        self._chat.insert("end", "▌", "cursor")
        self._chat.configure(state="disabled")
        self._chat.see("end")
        self._blink_cursor()

    def _on_token(self, token: str) -> None:
        """Append a streamed token — insert before the cursor glyph."""
        if not self._streaming:
            return
        self._chat.configure(state="normal")
        try:
            cursor_pos = self._chat.search("▌", "1.0", "end")
            if cursor_pos:
                self._chat.delete(cursor_pos)
                self._chat.insert(cursor_pos, token, "stream")
                # Re-insert cursor after the new text
                new_cursor = self._chat.index(f"{cursor_pos}+{len(token)}c")
                self._chat.insert(new_cursor, "▌", "cursor")
            else:
                self._chat.insert("end", token, "stream")
        except Exception:
            self._chat.insert("end", token, "stream")
        self._chat.configure(state="disabled")
        self._chat.see("end")

    def _on_done(self, full_text: str) -> None:
        """Stream complete — remove cursor glyph, add separator, store reply."""
        self._streaming = False
        self._send_btn.configure(state="normal", text=f"{G.EXEC}  Ask")

        self._chat.configure(state="normal")
        try:
            cursor_pos = self._chat.search("▌", "1.0", "end")
            if cursor_pos:
                self._chat.delete(cursor_pos)
        except Exception:
            pass
        self._chat.insert("end", f"\n{'─'*56}\n", "sep")
        self._chat.configure(state="disabled")
        self._chat.see("end")

        if full_text:
            self._history.append({"role": "assistant", "content": full_text})

    def _on_err(self, msg: str) -> None:
        self._streaming = False
        self._send_btn.configure(state="normal", text=f"{G.EXEC}  Ask")

        # Remove any partial streamed content back to the stream mark
        self._chat.configure(state="normal")
        try:
            start = self._chat.index(self._STREAM_MARK)
            self._chat.delete(start, "end")
        except Exception:
            pass
        self._chat.configure(state="disabled")
        self._append_err(f"Error: {msg}")

    def _blink_cursor(self) -> None:
        """Toggle cursor glyph colour every 500ms while streaming."""
        if not self._streaming:
            return
        self._chat.configure(state="normal")
        try:
            pos = self._chat.search("▌", "1.0", "end")
            if pos:
                end_pos = f"{pos}+1c"
                tags = self._chat.tag_names(pos)
                if "cursor" in tags:
                    self._chat.tag_remove("cursor", pos, end_pos)
                    self._chat.tag_add("dim", pos, end_pos)
                else:
                    self._chat.tag_remove("dim", pos, end_pos)
                    self._chat.tag_add("cursor", pos, end_pos)
        except Exception:
            pass
        self._chat.configure(state="disabled")
        if self._streaming:
            self.after(500, self._blink_cursor)

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _show_welcome(self) -> None:
        self._chat.configure(state="normal")
        self._chat.insert("end", "\n")
        self._chat.insert("end", f"  {G.SUN} ORACLE\n", "ai_lbl")
        self._chat.insert("end", (
            "Online — powered by Ollama (local AI, no internet, no API key).\n"
            "Responses stream live, token by token — no more waiting in silence.\n\n"
            "I have your live portfolio loaded. Ask me anything:\n\n"
            "  • Analyze my open positions and their risk\n"
            "  • What's NVDA's outlook right now?\n"
            "  • Why might the agent go long on AMD?\n"
            "  • Explain the Kelly criterion in simple terms\n\n"
            f"  {G.DOT_DIM}  Make sure Ollama.app is running (menu bar icon).\n"
            f"  {G.DOT_DIM}  First response may take up to 2 min on cold start.\n"
            f"  {G.DOT_DIM}  Click the  ↺  dot above if models aren't loading.\n"
        ), "ai_txt")
        self._chat.insert("end", f"\n{'─'*56}\n", "sep")
        self._chat.configure(state="disabled")

    def _append_you(self, text: str) -> None:
        self._chat.configure(state="normal")
        self._chat.insert("end", "\n")
        self._chat.insert("end",
            f"  YOU  {datetime.now().strftime('%H:%M')}\n", "you_lbl")
        self._chat.insert("end", f"{text}\n", "you_txt")
        self._chat.configure(state="disabled")
        self._chat.see("end")

    def _append_err(self, text: str) -> None:
        self._chat.configure(state="normal")
        self._chat.insert("end", f"\n  {G.SKULL}  {text}\n", "err")
        self._chat.configure(state="disabled")
        self._chat.see("end")

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
        if self._streaming:
            return
        self._history.clear()
        self._chat.configure(state="normal")
        self._chat.delete("1.0", "end")
        self._chat.insert("end", "\n")
        self._chat.insert("end", f"  {G.SUN} ORACLE\n", "ai_lbl")
        self._chat.insert("end",
            "Conversation cleared. Portfolio context still active.\n", "ai_txt")
        self._chat.insert("end", f"\n{'─'*56}\n", "sep")
        self._chat.configure(state="disabled")

    # ------------------------------------------------------------------
    # Live update hook
    # ------------------------------------------------------------------

    def update_from(self, snap: Dict[str, Any]) -> None:
        self._snap = snap
        self._refresh_ctx()
