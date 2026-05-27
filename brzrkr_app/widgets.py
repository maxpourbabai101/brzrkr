"""Themed widget library — gothic counterparts to plain customtkinter widgets.

Each widget is preset with the BRZRKR palette + fonts, so pages stay
short. Use :class:`GothicCard` as the container, :class:`SectionHeader`
for titled sub-sections, :class:`BloodMetric` for big numbers, and
:class:`RuneButton` for actions.
"""

from __future__ import annotations

from typing import Optional

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, FONT_SERIF


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
class GothicCard(ctk.CTkFrame):
    """Charcoal panel with a thin blood-toned border."""

    def __init__(self, parent, **kw):
        kw.setdefault("fg_color", C.PANEL)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 4)   # sharp, not rounded — gothic
        super().__init__(parent, **kw)


class IronFrame(ctk.CTkFrame):
    """Subtle separator frame, no border."""

    def __init__(self, parent, **kw):
        kw.setdefault("fg_color", "transparent")
        kw.setdefault("corner_radius", 0)
        super().__init__(parent, **kw)


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------
class SectionHeader(ctk.CTkFrame):
    """Banner-style header: ❰ ✠ TITLE ✠ ❱"""

    def __init__(self, parent, title: str, *, glyph: str = G.CROSS, **kw):
        kw.setdefault("fg_color", "transparent")
        super().__init__(parent, **kw)
        text = f"{G.SIGIL_L}  {glyph}  {title.upper()}  {glyph}  {G.SIGIL_R}"
        self._label = ctk.CTkLabel(
            self, text=text,
            text_color=C.SIGIL,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=12, weight="bold"),
            anchor="w",
        )
        self._label.pack(anchor="w", padx=14, pady=(12, 4))
        # Hairline divider beneath the heading.
        sep = ctk.CTkFrame(self, height=1, fg_color=C.BORDER)
        sep.pack(fill="x", padx=14, pady=(0, 8))


class PageTitle(ctk.CTkFrame):
    """Big page title with ornate side flourishes."""

    def __init__(self, parent, title: str, subtitle: str = "", **kw):
        kw.setdefault("fg_color", "transparent")
        super().__init__(parent, **kw)
        self.grid_columnconfigure(0, weight=1)
        line = ctk.CTkLabel(
            self,
            text=f"{G.ORNATE}   {title}   {G.ORNATE}",
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=22, weight="bold"),
            anchor="w",
        )
        line.grid(row=0, column=0, sticky="w", padx=4)
        if subtitle:
            sub = ctk.CTkLabel(
                self, text=subtitle.upper(),
                text_color=C.ASH,
                font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
                anchor="w",
            )
            sub.grid(row=1, column=0, sticky="w", padx=4, pady=(0, 4))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class BloodMetric(ctk.CTkFrame):
    """Big number metric with a small caption above and sub-line below."""

    def __init__(self, parent, caption: str, value: str = "—",
                 sub: str = "", color: Optional[str] = None, **kw):
        kw.setdefault("fg_color", C.PANEL)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 4)
        super().__init__(parent, **kw)
        self.grid_columnconfigure(0, weight=1)
        # Caption (small, ash, all caps)
        self.cap = ctk.CTkLabel(
            self, text=caption.upper(),
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
            anchor="w",
        )
        self.cap.grid(row=0, column=0, padx=16, pady=(14, 2), sticky="ew")
        # Value (big, serif/mono)
        self.val = ctk.CTkLabel(
            self, text=value,
            text_color=color or C.BONE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=26, weight="bold"),
            anchor="w",
        )
        self.val.grid(row=1, column=0, padx=16, pady=0, sticky="ew")
        # Sub (small, parchment)
        self.subl = ctk.CTkLabel(
            self, text=sub,
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_SERIF[0], size=11),
            anchor="w",
        )
        self.subl.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="ew")

    def set(self, value: str, *, sub: str = "", color: Optional[str] = None) -> None:
        self.val.configure(text=value, text_color=color or C.BONE)
        self.subl.configure(text=sub)


# ---------------------------------------------------------------------------
# Buttons
# ---------------------------------------------------------------------------
class RuneButton(ctk.CTkButton):
    """Primary action button: crimson on hover."""

    def __init__(self, parent, text: str, *, glyph: str = "", **kw):
        kw.setdefault("fg_color", C.BLOOD_DIM)
        kw.setdefault("hover_color", C.BLOOD)
        kw.setdefault("text_color", C.BONE)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 2)
        kw.setdefault("height", 34)
        kw.setdefault("font", ctk.CTkFont(family=FONT_SANS[0], size=12, weight="bold"))
        display_text = f"{glyph}  {text}" if glyph else text
        super().__init__(parent, text=display_text, **kw)


class GhostButton(ctk.CTkButton):
    """Subtle secondary button — outline only."""

    def __init__(self, parent, text: str, *, glyph: str = "", **kw):
        kw.setdefault("fg_color", "transparent")
        kw.setdefault("hover_color", C.PANEL_HI)
        kw.setdefault("text_color", C.PARCHMENT)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 2)
        kw.setdefault("height", 32)
        kw.setdefault("font", ctk.CTkFont(family=FONT_SANS[0], size=11))
        display_text = f"{glyph}  {text}" if glyph else text
        super().__init__(parent, text=display_text, **kw)


# ---------------------------------------------------------------------------
# Status indicators
# ---------------------------------------------------------------------------
class StatusBeacon(ctk.CTkFrame):
    """Glyph + label pair, color-coded to state."""

    def __init__(self, parent, label: str = "", state: str = "neutral",
                 glyph: str = G.DOT_ON, **kw):
        kw.setdefault("fg_color", "transparent")
        super().__init__(parent, **kw)
        from brzrkr_app.theme import state_color
        self._glyph_lbl = ctk.CTkLabel(
            self, text=glyph,
            text_color=state_color(state),
            font=ctk.CTkFont(family=FONT_SANS[0], size=14),
        )
        self._glyph_lbl.pack(side="left", padx=(0, 6))
        self._text_lbl = ctk.CTkLabel(
            self, text=label, text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_SANS[0], size=12, weight="bold"),
        )
        self._text_lbl.pack(side="left")

    def set(self, label: str, state: str = "neutral", glyph: Optional[str] = None) -> None:
        from brzrkr_app.theme import state_color, state_glyph
        self._glyph_lbl.configure(
            text=glyph or state_glyph(state),
            text_color=state_color(state),
        )
        self._text_lbl.configure(text=label)


class PulseBar(ctk.CTkFrame):
    """Horizontal activity bar — like a heartbeat / EKG. Pure tk canvas
    so we get pixel control. Updates via .pulse(value)."""

    def __init__(self, parent, width: int = 240, height: int = 18,
                 *, color: str = C.BLOOD, idle_color: str = C.IRON, **kw):
        kw.setdefault("fg_color", "transparent")
        super().__init__(parent, **kw)
        import tkinter as tk
        self._width = width
        self._height = height
        self._color = color
        self._idle = idle_color
        self._canvas = tk.Canvas(self, width=width, height=height,
                                  bg=C.OBSIDIAN, highlightthickness=0)
        self._canvas.pack()
        self._history = [0.0] * 60
        self._render()

    def pulse(self, value: float) -> None:
        """value in [0, 1]"""
        value = max(0.0, min(1.0, value))
        self._history.append(value)
        if len(self._history) > 60:
            self._history.pop(0)
        self._render()

    def _render(self) -> None:
        """Render the bar. Named _render (not _draw) to avoid colliding
        with CustomTkinter's internal CTkFrame._draw lifecycle method."""
        self._canvas.delete("all")
        n = len(self._history)
        bar_w = max(1, self._width / n)
        for i, v in enumerate(self._history):
            h = int(self._height * v)
            if h < 1:
                # Idle dot at bottom
                self._canvas.create_line(
                    i * bar_w, self._height - 1, (i + 1) * bar_w, self._height - 1,
                    fill=self._idle,
                )
                continue
            self._canvas.create_rectangle(
                i * bar_w, self._height - h, (i + 1) * bar_w, self._height,
                fill=self._color, outline="",
            )


# ---------------------------------------------------------------------------
# Text-display (for log tails, signal JSON, etc.)
# ---------------------------------------------------------------------------
class CodexBox(ctk.CTkTextbox):
    """Read-only monospace text panel with parchment styling."""

    def __init__(self, parent, **kw):
        kw.setdefault("fg_color", C.OBSIDIAN)
        kw.setdefault("text_color", C.PARCHMENT)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("font", ctk.CTkFont(family=FONT_MONO[0], size=11))
        kw.setdefault("wrap", "none")
        kw.setdefault("corner_radius", 2)
        super().__init__(parent, **kw)
        self.configure(state="disabled")

    def set_text(self, text: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert("1.0", text)
        self.configure(state="disabled")


# ---------------------------------------------------------------------------
# Mini candlestick chart — for the backtest grid
# ---------------------------------------------------------------------------
class MiniCandleChart(ctk.CTkFrame):
    """Compact candlestick chart, no axes — ideal for 2x2 grids.
    Set data via :meth:`set_data(df, title, subtitle)`."""

    def __init__(self, parent, *, height: int = 180, **kw):
        kw.setdefault("fg_color", C.PANEL)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 2)
        super().__init__(parent, **kw)
        self._height = height
        self._df = None
        self._title = ""
        self._subtitle = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))
        head.grid_columnconfigure(0, weight=1)
        self._title_lbl = ctk.CTkLabel(
            head, text="—",
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=12, weight="bold"),
            anchor="w",
        )
        self._title_lbl.grid(row=0, column=0, sticky="w")
        self._sub_lbl = ctk.CTkLabel(
            head, text="",
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
            anchor="e",
        )
        self._sub_lbl.grid(row=0, column=1, sticky="e")

        import tkinter as tk
        self._canvas = tk.Canvas(
            self, bg=C.SUMI, highlightthickness=0, height=height,
        )
        self._canvas.grid(row=1, column=0, padx=8, pady=(4, 8), sticky="nsew")
        self._canvas.bind("<Configure>", lambda e: self._render())

    def set_data(self, df, title: str = "", subtitle: str = "") -> None:
        self._df = df
        self._title = title
        self._subtitle = subtitle
        self._title_lbl.configure(text=title or "—")
        self._sub_lbl.configure(text=subtitle, text_color=C.ASH)
        self._render()

    def _render(self) -> None:
        c = self._canvas
        c.delete("all")
        df = self._df
        if df is None or len(df) == 0:
            w = c.winfo_width() or 200
            h = c.winfo_height() or 100
            c.create_text(w // 2, h // 2,
                           text=f"{G.MOON}  no data",
                           fill=C.ASH,
                           font=(FONT_DISPLAY[0], 11, "bold"))
            return

        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or h < 20:
            return

        ml, mr, mt, mb = 6, 6, 6, 6
        pw, ph = w - ml - mr, h - mt - mb

        try:
            hi = float(df["high"].max())
            lo = float(df["low"].min())
        except Exception:
            return
        if hi == lo:
            hi += 1

        def y(p):
            return mt + ph - (p - lo) / (hi - lo) * ph

        n = len(df)
        slot = pw / max(n, 1)
        bar_w = max(1.0, slot * 0.7)

        for i, (_, row) in enumerate(df.iterrows()):
            x = ml + (i + 0.5) * slot
            try:
                o = float(row["open"]); cl = float(row["close"])
                hi_p = float(row["high"]); lo_p = float(row["low"])
            except Exception:
                continue
            up = cl >= o
            color = C.LIFE if up else C.WOUND
            c.create_line(x, y(hi_p), x, y(lo_p), fill=color, width=1)
            top = y(max(o, cl)); bot = y(min(o, cl))
            if abs(bot - top) < 1: bot = top + 1
            c.create_rectangle(x - bar_w / 2, top, x + bar_w / 2, bot,
                                fill=color, outline=color)


# ---------------------------------------------------------------------------
# Sci-fi circular gauge — for the System tab
# ---------------------------------------------------------------------------
class SciFiGauge(ctk.CTkFrame):
    """Circular gauge with an arc fill, animated as you call .set(value).
    Shows label + value + max unit. Movie-grade."""

    def __init__(self, parent, label: str, *,
                 max_value: float = 100.0, unit: str = "%",
                 size: int = 140, color: str | None = None, **kw):
        kw.setdefault("fg_color", C.PANEL)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 2)
        super().__init__(parent, **kw)
        self._label = label
        self._unit = unit
        self._max = max_value
        self._color = color or C.WOUND
        self._size = size
        self._value = 0.0

        self.grid_columnconfigure(0, weight=1)

        import tkinter as tk
        self._canvas = tk.Canvas(
            self, width=size, height=size,
            bg=C.PANEL, highlightthickness=0,
        )
        self._canvas.grid(row=0, column=0, padx=16, pady=(16, 4))

        self._label_lbl = ctk.CTkLabel(
            self, text=label.upper(),
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
        )
        self._label_lbl.grid(row=1, column=0, pady=(0, 14))

        self._render()

    def set(self, value: float, *, label: str | None = None) -> None:
        self._value = max(0.0, min(self._max, value))
        if label:
            self._label_lbl.configure(text=label.upper())
        self._render()

    def _render(self) -> None:
        c = self._canvas
        c.delete("all")
        s = self._size
        pad = 12
        cx = cy = s // 2
        r = s // 2 - pad

        # Outer decorative ring (iron)
        c.create_oval(pad - 4, pad - 4, s - pad + 4, s - pad + 4,
                       outline=C.IRON, width=1)
        # Inner blood ring
        c.create_oval(pad, pad, s - pad, s - pad,
                       outline=C.BLOOD_DIM, width=2)

        # Background arc (full circle, faint)
        c.create_arc(pad + 8, pad + 8, s - pad - 8, s - pad - 8,
                      start=135, extent=270,
                      style="arc", outline=C.INK_HI, width=10)

        # Foreground arc (proportional)
        frac = self._value / self._max if self._max > 0 else 0
        extent = 270 * frac
        # Color shifts toward FORGE as value rises
        col = self._color
        if frac > 0.85:
            col = C.FORGE
        elif frac > 0.7:
            col = C.EMBER
        c.create_arc(pad + 8, pad + 8, s - pad - 8, s - pad - 8,
                      start=135, extent=-extent,    # negative = clockwise
                      style="arc", outline=col, width=10)

        # Center text
        val_text = f"{self._value:.0f}" if self._max >= 10 else f"{self._value:.1f}"
        c.create_text(cx, cy - 4, text=val_text,
                       fill=C.BONE,
                       font=(FONT_MONO[0], 24, "bold"))
        c.create_text(cx, cy + 22, text=self._unit,
                       fill=C.PARCHMENT,
                       font=(FONT_SANS[0], 10, "bold"))


# ---------------------------------------------------------------------------
# Horizontal bar meter — for per-core CPU, network rates, etc.
# ---------------------------------------------------------------------------
class BarMeter(ctk.CTkFrame):
    """Label + horizontal bar + value. Lightweight, draws via canvas."""

    def __init__(self, parent, label: str = "", *,
                 width: int = 240, height: int = 18,
                 max_value: float = 100.0, unit: str = "%",
                 color: str | None = None, **kw):
        kw.setdefault("fg_color", "transparent")
        super().__init__(parent, **kw)
        self._max = max_value
        self._unit = unit
        self._color = color or C.WOUND
        self._height = height
        self._width = width
        self._value = 0.0

        self.grid_columnconfigure(1, weight=1)

        self._lbl = ctk.CTkLabel(
            self, text=label,
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
            anchor="w", width=60,
        )
        self._lbl.grid(row=0, column=0, sticky="w", padx=(0, 6))

        import tkinter as tk
        self._canvas = tk.Canvas(
            self, width=width, height=height,
            bg=C.INK_DEEP, highlightthickness=0,
        )
        self._canvas.grid(row=0, column=1, sticky="ew")

        self._value_lbl = ctk.CTkLabel(
            self, text="—",
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10, weight="bold"),
            anchor="e", width=50,
        )
        self._value_lbl.grid(row=0, column=2, sticky="e", padx=(6, 0))

        self._render()

    def set(self, value: float, *, label: str | None = None) -> None:
        self._value = max(0.0, min(self._max, value))
        if label:
            self._lbl.configure(text=label)
        self._render()

    def _render(self) -> None:
        c = self._canvas
        c.delete("all")
        w = self._canvas.winfo_reqwidth() or self._width
        h = self._height
        # Background grid lines (10%)
        for i in range(1, 10):
            x = w * i / 10
            c.create_line(x, 0, x, h, fill=C.IRON, width=1)
        # Filled portion
        frac = self._value / self._max if self._max > 0 else 0
        col = self._color
        if frac > 0.85: col = C.FORGE
        elif frac > 0.7: col = C.EMBER
        c.create_rectangle(0, 0, w * frac, h, fill=col, outline="")
        # Border
        c.create_rectangle(0, 0, w, h, outline=C.BORDER)

        self._value_lbl.configure(text=f"{self._value:.0f}{self._unit}",
                                    text_color=col)


# ---------------------------------------------------------------------------
# Equity curve — for Strategy tab
# ---------------------------------------------------------------------------
class EquityCurve(ctk.CTkFrame):
    """Line chart of equity over time. Set via .set_data(series)."""

    def __init__(self, parent, *, height: int = 220, **kw):
        kw.setdefault("fg_color", C.PANEL)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 2)
        super().__init__(parent, **kw)
        self._height = height
        self._values: list[float] = []
        self._labels: list[str] = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        import tkinter as tk
        self._canvas = tk.Canvas(
            self, height=height, bg=C.SUMI, highlightthickness=0,
        )
        self._canvas.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self._canvas.bind("<Configure>", lambda e: self._render())

    def set_data(self, values, labels=None) -> None:
        self._values = [float(v) for v in values]
        self._labels = list(labels) if labels else []
        self._render()

    def _render(self) -> None:
        c = self._canvas
        c.delete("all")
        if not self._values:
            w = c.winfo_width() or 400
            h = c.winfo_height() or 200
            c.create_text(w // 2, h // 2,
                           text=f"{G.DOT_DIM}  no sessions yet",
                           fill=C.ASH,
                           font=(FONT_DISPLAY[0], 13, "bold"))
            return

        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20 or h < 20: return
        ml, mr, mt, mb = 50, 12, 14, 22
        pw = w - ml - mr; ph = h - mt - mb

        hi = max(self._values); lo = min(self._values)
        if hi == lo: hi += 1
        pad = (hi - lo) * 0.05
        hi += pad; lo -= pad

        def y(v): return mt + ph - (v - lo) / (hi - lo) * ph

        # Grid lines
        for i in range(5):
            yv = mt + ph * i / 4
            val = hi - (hi - lo) * i / 4
            c.create_line(ml, yv, ml + pw, yv, fill=C.IRON)
            c.create_text(ml - 6, yv, anchor="e",
                           text=f"${val:,.0f}", fill=C.ASH,
                           font=(FONT_MONO[0], 9))

        # Determine if final value is up or down from start
        is_up = self._values[-1] >= self._values[0]
        line_col = C.LIFE if is_up else C.WOUND

        # Fill area under curve (subtle)
        n = len(self._values)
        if n >= 2:
            pts = []
            for i, v in enumerate(self._values):
                x = ml + (i / (n - 1)) * pw if n > 1 else ml
                pts.extend([x, y(v)])
            # Fill polygon
            poly_pts = [ml, mt + ph] + pts + [ml + pw, mt + ph]
            c.create_polygon(poly_pts, fill=C.INK_HI, outline="")
            # Line on top
            c.create_line(*pts, fill=line_col, width=2, smooth=False)
            # Dots at each point
            for i in range(n):
                x = ml + (i / (n - 1)) * pw if n > 1 else ml
                c.create_oval(x - 2, y(self._values[i]) - 2,
                               x + 2, y(self._values[i]) + 2,
                               fill=line_col, outline="")

        # Border
        c.create_rectangle(ml, mt, ml + pw, mt + ph,
                            outline=C.BORDER)
        # X-axis labels (first/last only)
        if self._labels:
            c.create_text(ml, mt + ph + 12, anchor="w",
                           text=self._labels[0], fill=C.ASH,
                           font=(FONT_MONO[0], 9))
            c.create_text(ml + pw, mt + ph + 12, anchor="e",
                           text=self._labels[-1], fill=C.ASH,
                           font=(FONT_MONO[0], 9))


# ---------------------------------------------------------------------------
# LiveSimCard — one slot in a parallel-sim grid
# ---------------------------------------------------------------------------
class LiveSimCard(ctk.CTkFrame):
    """A panel that shows a single live simulation slot: scenario name,
    progress bar, mini equity curve, current P/L.

    Reads from a per-slot status dict (or None if the slot is idle).
    Designed to be tiled into a 2×2 grid by the Backtests page.
    """

    def __init__(self, parent, *, slot: int, height: int = 240, **kw):
        kw.setdefault("fg_color", C.PANEL)
        kw.setdefault("border_color", C.BORDER)
        kw.setdefault("border_width", 1)
        kw.setdefault("corner_radius", 4)
        super().__init__(parent, **kw)
        self._slot = slot

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Header: slot number + beacon + scenario title
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        head.grid_columnconfigure(1, weight=1)
        self._slot_lbl = ctk.CTkLabel(
            head, text=f"{G.RUNE_T}  SLOT {slot + 1}",
            text_color=C.SIGIL,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=11, weight="bold"),
        )
        self._slot_lbl.grid(row=0, column=0, sticky="w")
        self._title_lbl = ctk.CTkLabel(
            head, text="—  idle  —",
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=13, weight="bold"),
            anchor="e",
        )
        self._title_lbl.grid(row=0, column=1, sticky="e")

        # Progress bar
        self._bar = BarMeter(self, label="bars", width=320,
                              max_value=100, unit="%")
        self._bar.grid(row=1, column=0, padx=12, pady=(4, 4), sticky="ew")

        # Mini equity curve
        self._curve = EquityCurve(self, height=110)
        self._curve.grid(row=2, column=0, padx=8, pady=(4, 4), sticky="nsew")

        # Footer: equity + P/L
        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 10))
        foot.grid_columnconfigure((0, 1, 2), weight=1)

        self._equity_lbl = ctk.CTkLabel(
            foot, text="equity\n—",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
            anchor="w", justify="left",
        )
        self._equity_lbl.grid(row=0, column=0, sticky="w")
        self._pl_lbl = ctk.CTkLabel(
            foot, text="P/L\n—",
            text_color=C.BONE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11, weight="bold"),
            anchor="center", justify="center",
        )
        self._pl_lbl.grid(row=0, column=1, sticky="ew")
        self._trades_lbl = ctk.CTkLabel(
            foot, text="trades\n—",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
            anchor="e", justify="right",
        )
        self._trades_lbl.grid(row=0, column=2, sticky="e")

    def update_from(self, status: dict | None) -> None:
        from brzrkr_app.theme import pnl_color
        if status is None:
            self._title_lbl.configure(text="—  idle  —", text_color=C.ASH)
            self._bar.set(0)
            self._curve.set_data([])
            self._equity_lbl.configure(text="equity\n—")
            self._pl_lbl.configure(text="P/L\n—", text_color=C.ASH)
            self._trades_lbl.configure(text="trades\n—")
            return

        active = bool(status.get("active"))
        scen = status.get("scenario") or "—"
        sym = status.get("symbol") or "—"
        cat = status.get("category") or ""
        bars_done = int(status.get("bars_processed", 0))
        bars_total = max(1, int(status.get("bars_total", 1)))
        equity = float(status.get("current_equity", 0.0))
        initial = float(status.get("initial_equity", 0.0)) or 100_000.0
        history = list(status.get("equity_history") or [])
        trades = int(status.get("trades_so_far", 0))

        title_color = C.BONE if active else C.ASH
        marker = G.RUNE_F if active else G.DOT_DIM
        self._title_lbl.configure(
            text=f"{marker}  {scen[:22]} × {sym}",
            text_color=title_color,
        )
        scen_pct = (bars_done / bars_total) * 100 if bars_total else 0
        self._bar.set(scen_pct)
        self._curve.set_data(history)

        pl = equity - initial
        pl_pct = (pl / initial * 100) if initial else 0
        self._equity_lbl.configure(text=f"equity\n${equity:,.0f}")
        self._pl_lbl.configure(
            text=f"P/L\n${pl:+,.0f}  ({pl_pct:+.2f}%)",
            text_color=pnl_color(pl),
        )
        self._trades_lbl.configure(text=f"trades\n{trades}")


# ---------------------------------------------------------------------------
# Ink-brush divider — vagabond/sumi-e horizontal rule
# ---------------------------------------------------------------------------
class InkDivider(ctk.CTkFrame):
    """A torn/brushed horizontal divider, drawn with canvas strokes."""

    def __init__(self, parent, *, length: int = 400, color: str | None = None,
                 **kw):
        kw.setdefault("fg_color", "transparent")
        super().__init__(parent, **kw)
        import tkinter as tk
        h = 10
        self._canvas = tk.Canvas(self, width=length, height=h,
                                  bg=C.NIGHT, highlightthickness=0)
        self._canvas.pack()
        col = color or C.BRUSH_RED
        # Main brush stroke — variable width
        import random
        random.seed(length)
        y = h // 2
        for i in range(8):
            x1 = i * length // 8
            x2 = (i + 1) * length // 8
            offset = random.choice([-1, 0, 1])
            w = random.choice([1, 2, 1])
            self._canvas.create_line(x1, y + offset, x2, y + offset,
                                       fill=col, width=w)
        # A few small splatter dots
        for _ in range(3):
            x = random.randint(20, length - 20)
            self._canvas.create_oval(x, y - 1, x + 1, y + 1,
                                       fill=col, outline="")
