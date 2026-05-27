"""Postmortem Codex — browse, search, and inspect the failure-mode DB."""

from __future__ import annotations

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS
from brzrkr_app.widgets import (
    CodexBox, GhostButton, GothicCard, PageTitle, RuneButton, SectionHeader,
)


class PostmortemPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app
        self._lessons = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(3, weight=1)

        PageTitle(self, f"{G.SKULL} Postmortem Codex",
                   subtitle="failure modes · self-confirmed").grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        # ---- Explanation header -----------------------------------
        intro = GothicCard(self)
        intro.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 12))
        intro.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            intro,
            text=(
                f"  {G.CROSS}  WHY THIS EXISTS\n"
                "  ─────────────────────────────────────────────────────────────────\n"
                "  Every quant strategy that ever blew up did so for a reason that\n"
                "  someone else had already documented. The Postmortem Codex is the\n"
                "  system's running list of those reasons — bundled with 52 lessons\n"
                "  from real quant history (LTCM, Knight Capital, GameStop squeeze,\n"
                "  Volmageddon, momentum crash, etc.) and growing automatically as\n"
                "  the agent observes its own behavior.\n\n"
                f"  {G.RUNE_T}  WHAT EACH LESSON CONTAINS\n"
                "  ─────────────────────────────────────────────────────────────────\n"
                "    • category (overfitting, leakage, execution, behavioral, …)\n"
                "    • severity 1–5  (5 = account-ending)\n"
                "    • symptom  — how to detect it in this system\n"
                "    • mitigation — what to do about it\n"
                "    • confirmed × N — how many times the observer has matched it\n\n"
                f"  {G.RUNE_F}  HOW IT'S USED\n"
                "  ─────────────────────────────────────────────────────────────────\n"
                "    • Preflight: surfaces relevant high-severity lessons before\n"
                "      every agent or training run; blocks startup for unmitigated\n"
                "      conditions (e.g. live money without a track record).\n"
                "    • Observer:  after every session, matches behavior against\n"
                "      known patterns and adds new ones for novel failure modes.\n"
                "    • Analyzer:  click 'Run Analysis' below — measures whether\n"
                "      lessons that fire correlate with actual P/L losses in YOUR\n"
                "      data, and promotes/demotes their severity accordingly."
            ),
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=11),
            justify="left", anchor="w",
        ).grid(row=0, column=0, padx=14, pady=14, sticky="w")

        # Filter bar
        bar = GothicCard(self)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        bar.grid_columnconfigure(0, weight=1)
        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="ew", padx=14, pady=12)
        inner.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(inner, text="CATEGORY",
                      text_color=C.PARCHMENT,
                      font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold")
                      ).grid(row=0, column=0, padx=(0, 8))
        self.v_cat = ctk.StringVar(value="all")
        ctk.CTkOptionMenu(inner,
                           values=["all", "overfitting", "leakage",
                                    "regime_shift", "execution", "risk_sizing",
                                    "correlation", "liquidity", "deployment",
                                    "behavioral", "data_quality",
                                    "counterparty", "macro", "crowding"],
                           variable=self.v_cat, width=160,
                           fg_color=C.PANEL_HI,
                           button_color=C.BLOOD_DIM,
                           button_hover_color=C.BLOOD,
                           text_color=C.BONE,
                           dropdown_fg_color=C.PANEL,
                           command=lambda _v: self._refresh()
                           ).grid(row=0, column=1)

        ctk.CTkLabel(inner, text="MIN SEVERITY",
                      text_color=C.PARCHMENT,
                      font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold")
                      ).grid(row=0, column=2, padx=(20, 8))
        self.v_sev = ctk.StringVar(value="1")
        ctk.CTkOptionMenu(inner, values=["1", "2", "3", "4", "5"],
                           variable=self.v_sev, width=70,
                           fg_color=C.PANEL_HI,
                           button_color=C.BLOOD_DIM,
                           button_hover_color=C.BLOOD,
                           text_color=C.BONE,
                           dropdown_fg_color=C.PANEL,
                           command=lambda _v: self._refresh()
                           ).grid(row=0, column=3, sticky="w")

        RuneButton(inner, "Run Analysis", glyph=G.RUNE_T,
                    command=self._analyze
                    ).grid(row=0, column=4, padx=(20, 0))

        # Lesson list (left)
        list_card = GothicCard(self)
        list_card.grid(row=3, column=0, sticky="nsew", padx=(0, 6))
        list_card.grid_columnconfigure(0, weight=1)
        list_card.grid_rowconfigure(1, weight=1)
        SectionHeader(list_card, "Lessons", glyph=G.CROSS).grid(
            row=0, column=0, sticky="ew")
        self.list_scroll = ctk.CTkScrollableFrame(list_card, fg_color="transparent")
        self.list_scroll.grid(row=1, column=0, padx=10, pady=(0, 14), sticky="nsew")
        self.list_scroll.grid_columnconfigure(0, weight=1)

        # Detail view (right)
        det_card = GothicCard(self)
        det_card.grid(row=3, column=1, sticky="nsew", padx=(6, 0))
        det_card.grid_columnconfigure(0, weight=1)
        det_card.grid_rowconfigure(1, weight=1)
        SectionHeader(det_card, "Detail", glyph=G.RUNE_O).grid(
            row=0, column=0, sticky="ew")
        self.detail_box = CodexBox(det_card)
        self.detail_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

        self._refresh()

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
        # Lessons change over time; refresh every poll cycle.
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        try:
            from src.learning.postmortem_db import PostmortemDB
            db = PostmortemDB()
            db.bootstrap_if_empty()
            cat = self.v_cat.get()
            sev = int(self.v_sev.get())
            kw = {"min_severity": sev}
            if cat != "all":
                kw["category"] = cat
            self._lessons = db.find(**kw)
        except Exception as exc:  # noqa: BLE001
            self._lessons = []

        for c in self.list_scroll.winfo_children():
            c.destroy()
        if not self._lessons:
            ctk.CTkLabel(self.list_scroll, text=f"{G.DOT_DIM}  none match",
                          text_color=C.ASH).grid(row=0, column=0, padx=4, pady=8, sticky="w")
            return
        for i, l in enumerate(self._lessons):
            sev_txt = G.DOT_ON * l.severity + G.DOT_OFF * (5 - l.severity)
            txt = f"{sev_txt}  [{l.id:>6s}]  {l.title[:40]}"
            btn = ctk.CTkButton(
                self.list_scroll, text=txt, anchor="w",
                fg_color="transparent",
                hover_color=C.PANEL_HI,
                text_color=C.BONE,
                font=ctk.CTkFont(family=FONT_MONO[0], size=11),
                command=lambda lid=l.id: self._show(lid),
                height=24, corner_radius=2,
            )
            btn.grid(row=i, column=0, sticky="ew", pady=1)

    def _show(self, lesson_id: str) -> None:
        try:
            from src.learning.postmortem_db import PostmortemDB
            l = PostmortemDB().get(lesson_id)
        except Exception:
            l = None
        if l is None:
            self.detail_box.set_text(f"{G.SKULL}  Lesson {lesson_id} not found.")
            return
        sev_str = G.DOT_ON * l.severity + G.DOT_OFF * (5 - l.severity)
        text = (
            f"  {G.CROSS}  {l.title}\n"
            f"  {'─' * 60}\n\n"
            f"  id          : {l.id}\n"
            f"  category    : {l.category}\n"
            f"  severity    : {sev_str}  ({l.severity}/5)\n"
            f"  source      : {l.source}\n"
            f"  confirmed   : {l.confirmed_count}× by observer\n"
            f"  first seen  : {l.first_seen}\n"
            f"  last confirm: {l.last_confirmed or '—'}\n"
        )
        if l.tags:
            text += f"  tags        : {', '.join(l.tags)}\n"
        text += (
            f"\n  {G.ORNATE}  DESCRIPTION\n  {'─' * 60}\n"
            f"  {l.description}\n\n"
            f"  {G.ORNATE}  SYMPTOM\n  {'─' * 60}\n"
            f"  {l.symptom}\n\n"
            f"  {G.ORNATE}  MITIGATION\n  {'─' * 60}\n"
            f"  {l.mitigation}\n"
        )
        if l.references:
            text += f"\n  {G.ORNATE}  REFERENCES\n  {'─' * 60}\n"
            for r in l.references:
                text += f"  • {r}\n"
        self.detail_box.set_text(text)

    def _analyze(self) -> None:
        """Run the correlation analyzer and display its report."""
        try:
            from src.learning.correlation_analyzer import CorrelationAnalyzer
            report = CorrelationAnalyzer().analyze()
            self.detail_box.set_text(report.to_text())
            self.app.toast(f"Analyzed {report.n_sessions} sessions.")
        except Exception as exc:  # noqa: BLE001
            self.detail_box.set_text(f"{G.SKULL}  Analysis failed: {exc}")
