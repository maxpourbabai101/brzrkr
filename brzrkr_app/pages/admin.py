"""Admin — project map, activity meters, code browser.

Two halves:

* **Top half**: metric strip + Living Indicators (pulse bars), as before.
* **Bottom half**: a categorized **code browser**. Files are split into
  domains (Scripts, Data, Model, Risk, Execution, Learning, etc.).
  Pick a category from the tab strip, click a file to view its source
  in the right pane. Nothing hidden, nothing messy.

Plus Inscribe buttons (system report, project tree, postmortem MD export).
"""

from __future__ import annotations

import json
import os
import tkinter.ttk as ttk
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS
from brzrkr_app.widgets import (
    BloodMetric, CodexBox, GhostButton, GothicCard, InkDivider,
    PageTitle, PulseBar, RuneButton, SectionHeader,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SRC_DIR = ROOT / "src"


# ---------------------------------------------------------------------------
# Categorized file map for the source-code browser
# ---------------------------------------------------------------------------
CODE_CATEGORIES: Dict[str, List[tuple]] = {
    "Scripts": [
        ("BRZRKR.py",           "main launcher  ·  opens this desktop app"),
        ("agent.py",            "autonomous trading loop"),
        ("trade.py",            "manual order CLI"),
        ("train.py",            "XGBoost training pipeline"),
        ("run.py",              "single-shot live / backtest entry"),
        ("learn.py",            "postmortem DB CLI"),
        ("weekend_practice.py", "historical scenario battery runner"),
        ("brz",                 "detached launcher (shell)"),
        ("BRZRKR.command",      "Dock-droppable macOS launcher"),
        ("build_brzrkr.sh",     "build a .app bundle"),
        ("setup.sh",            "create venv + install requirements"),
    ],
    "Data": [
        ("src/data_loader.py",       "vendor API fetchers (Alpaca/Tradier/NewsAPI/FRED)"),
        ("src/data_scraper.py",      "web scraping (Yahoo/SEC/Reddit/Congress)"),
        ("src/data_alternatives.py", "free options + sentiment fallback"),
    ],
    "Features": [
        ("src/features/feature_engineer.py", "features for ensemble + training"),
    ],
    "Model": [
        ("src/model/ensemble.py",             "weighted LSTM + XGB + Transformer"),
        ("src/model/transformer_backbone.py", "3-layer Transformer (seq=256)"),
        ("src/model/sentiment_encoder.py",    "FinBERT wrapper"),
        ("src/model/volatility_module.py",    "GARCH(1,1) + IV spline"),
        ("src/model/xgb_predictor.py",        "trained XGBoost inference"),
    ],
    "Risk": [
        ("src/risk/risk_manager.py",    "Kelly sizing, stops, filters"),
        ("src/risk/countermeasures.py", "circuit breaker, cooldown, regimes"),
    ],
    "Execution": [
        ("src/execution/broker.py",         "Alpaca paper/live executor"),
        ("src/execution/brokers.py",        "multi-broker registry"),
        ("src/execution/promotion_gate.py", "paper → live gate"),
    ],
    "Agent": [
        ("src/agent/trading_agent.py", "the autonomous loop class"),
    ],
    "Learning": [
        ("src/learning/postmortem_db.py",        "failure-mode knowledge base"),
        ("src/learning/seed_knowledge.py",       "52 seeded historical lessons"),
        ("src/learning/observer.py",             "session → DB updates"),
        ("src/learning/preflight.py",            "pre-trade safety checks"),
        ("src/learning/correlation_analyzer.py", "conditional-P&L analysis"),
    ],
    "Training": [
        ("src/training/feature_dataset.py", "labels + features for ML"),
        ("src/training/trainer.py",         "walk-forward XGBoost training"),
    ],
    "Backtest": [
        ("src/signals/signal_generator.py",  "signal JSON builder"),
        ("src/backtest/backtest_runner.py",  "single-run backtest engine"),
        ("src/backtest/scenario_runner.py",  "parallel scenario battery"),
        ("src/backtest/scenarios.py",        "21 documented market episodes"),
        ("src/backtest/live_status.py",      "live JSON status for sims"),
    ],
    "App (UI)": [
        ("brzrkr_app/main_window.py",      "main window + sidebar"),
        ("brzrkr_app/theme.py",            "palette + fonts + glyphs"),
        ("brzrkr_app/widgets.py",          "reusable themed widgets"),
        ("brzrkr_app/poller.py",           "broker polling thread"),
        ("brzrkr_app/icon.py",             "app icon generator (PIL)"),
        ("brzrkr_app/pages/status.py",     "Status tab"),
        ("brzrkr_app/pages/trades.py",     "Live Trades tab"),
        ("brzrkr_app/pages/console.py",    "Trade Console tab"),
        ("brzrkr_app/pages/market.py",     "Market Data tab"),
        ("brzrkr_app/pages/backtests.py",  "Backtests tab (live sims)"),
        ("brzrkr_app/pages/strategy.py",   "Strategy tab"),
        ("brzrkr_app/pages/postmortem.py", "Postmortem tab"),
        ("brzrkr_app/pages/system.py",     "System monitor tab"),
        ("brzrkr_app/pages/admin.py",      "this Admin tab"),
    ],
    "Utils": [
        ("src/utils/logging_setup.py", "rotating file + console logger"),
        ("src/utils/cron_job.sh",      "daily cron entry"),
    ],
    "Tests": [
        ("tests/test_data_loader.py",          "vendor API tests"),
        ("tests/test_data_scraper.py",         "scraper tests"),
        ("tests/test_data_alternatives.py",    "free alternatives"),
        ("tests/test_ensemble.py",             "ensemble aggregator"),
        ("tests/test_risk_manager.py",         "Kelly + stops + filters"),
        ("tests/test_countermeasures.py",      "circuit breaker etc."),
        ("tests/test_broker.py",               "Alpaca executor"),
        ("tests/test_brokers.py",              "registry + in-memory broker"),
        ("tests/test_promotion_gate.py",       "live-money gate"),
        ("tests/test_trading_agent.py",        "autonomous loop"),
        ("tests/test_feature_engineer.py",     "feature engineer"),
        ("tests/test_training.py",             "training pipeline"),
        ("tests/test_scenarios.py",            "scenario battery"),
        ("tests/test_live_status.py",          "live status writer"),
        ("tests/test_learning.py",             "postmortem DB"),
        ("tests/test_correlation_analyzer.py", "correlation analyzer"),
    ],
    "Config + Docs": [
        ("config/config.yaml",                "central config (thresholds, params)"),
        ("config/secrets.yaml",               "API key TEMPLATE (no real keys)"),
        ("requirements.txt",                  "pinned Python dependencies"),
        ("README.md",                         "project documentation"),
        ("docs/architecture.md",              "design notes"),
        ("docs/deployment.md",                "deployment guide"),
        ("docs/trading_strategy_sources.md",  "120+ research sources"),
    ],
}


class AdminPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app

        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C.NIGHT)
        self._scroll.pack(fill="both", expand=True)
        body = self._scroll
        body.grid_columnconfigure((0, 1, 2), weight=1)

        PageTitle(body, f"{G.DAGGER} Admin Crypt",
                   subtitle="program map · activity · source").grid(
            row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        InkDivider(body, length=720).grid(row=1, column=0, columnspan=3,
                                            sticky="w", pady=(0, 12))

        # Top metrics
        self.m_files = BloodMetric(body, "Source Files", "—")
        self.m_data = BloodMetric(body, "Data Artifacts", "—")
        self.m_models = BloodMetric(body, "Models", "—")
        self.m_files.grid(row=2, column=0, padx=(0, 6), sticky="nsew")
        self.m_data.grid(row=2, column=1, padx=6, sticky="nsew")
        self.m_models.grid(row=2, column=2, padx=(6, 0), sticky="nsew")

        # Living indicators
        act = GothicCard(body)
        act.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 12))
        act.grid_columnconfigure(0, weight=1)
        SectionHeader(act, "Living Indicators", glyph=G.RUNE_T).grid(
            row=0, column=0, sticky="ew")
        ind = ctk.CTkFrame(act, fg_color="transparent")
        ind.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        for i in range(4):
            ind.grid_columnconfigure(i, weight=1)

        def beacon(col, name, glyph):
            f = ctk.CTkFrame(ind, fg_color="transparent")
            f.grid(row=0, column=col, sticky="nsew", padx=4)
            ctk.CTkLabel(f, text=f"{glyph}  {name.upper()}",
                          text_color=C.PARCHMENT,
                          font=ctk.CTkFont(family=FONT_SANS[0], size=10, weight="bold"),
                          anchor="w").pack(anchor="w", pady=(0, 4))
            bar = PulseBar(f, width=200, height=16)
            bar.pack(anchor="w")
            return bar

        self.bar_log = beacon(0, "log activity", G.RUNE_F)
        self.bar_signals = beacon(1, "signal forge", G.RUNE_O)
        self.bar_orders = beacon(2, "order flow", G.EXEC)
        self.bar_lessons = beacon(3, "lesson db", G.RUNE_R)

        self._last_log_mtime = 0
        self._last_signal_count = 0
        self._last_order_count = 0
        self._last_lesson_count = 0

        # ---- CODE BROWSER ------------------------------------------
        browser = GothicCard(body)
        browser.grid(row=4, column=0, columnspan=3, sticky="nsew",
                      pady=(0, 12))
        browser.grid_columnconfigure(0, weight=1)
        browser.grid_rowconfigure(1, weight=1)
        SectionHeader(browser, "Codex of Source — entire project",
                       glyph=G.CROSS).grid(row=0, column=0, sticky="ew")

        self.tabview = ctk.CTkTabview(
            browser,
            segmented_button_fg_color=C.PANEL_HI,
            segmented_button_selected_color=C.BLOOD_DIM,
            segmented_button_selected_hover_color=C.BLOOD,
            segmented_button_unselected_color=C.PANEL,
            segmented_button_unselected_hover_color=C.PANEL_HI,
            text_color=C.BONE,
            fg_color=C.NIGHT,
            border_color=C.BORDER,
            border_width=1,
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=12,
                           pady=(0, 12))
        # Force the tabview to render large enough to fit the source pane.
        browser.configure(height=520)
        browser.grid_propagate(False)
        self._build_browser_tabs()

        # ---- INSCRIBE exports --------------------------------------
        exp = GothicCard(body)
        exp.grid(row=5, column=0, columnspan=3, sticky="ew")
        exp.grid_columnconfigure((0, 1, 2), weight=1)
        SectionHeader(exp, "Inscribe", glyph=G.RUNE_O).grid(
            row=0, column=0, columnspan=3, sticky="ew")
        RuneButton(exp, "System Report", glyph=G.RUNE_O,
                    command=self._inscribe).grid(row=1, column=0,
                                                    padx=14, pady=(4, 6),
                                                    sticky="ew")
        GhostButton(exp, "Project Tree", glyph=G.CROSS,
                     command=self._inscribe_map).grid(row=1, column=1,
                                                         padx=14, pady=(4, 6),
                                                         sticky="ew")
        GhostButton(exp, "Postmortem MD", glyph=G.SKULL,
                     command=self._inscribe_postmortem).grid(row=1, column=2,
                                                                padx=14, pady=(4, 6),
                                                                sticky="ew")
        self.exp_status = ctk.CTkLabel(
            exp, text="", text_color=C.LIFE,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
            anchor="w",
        )
        self.exp_status.grid(row=2, column=0, columnspan=3, padx=14,
                              pady=(0, 14), sticky="w")

    # ------------------------------------------------------------------
    def _build_browser_tabs(self) -> None:
        for cat_name, items in CODE_CATEGORIES.items():
            tab = self.tabview.add(cat_name)
            tab.grid_columnconfigure(1, weight=1)
            tab.grid_rowconfigure(0, weight=1)

            # Left: file list (scrollable)
            left = ctk.CTkScrollableFrame(
                tab, fg_color=C.PANEL,
                border_color=C.BORDER, border_width=1,
                corner_radius=2, width=300,
            )
            left.grid(row=0, column=0, sticky="nsw", padx=(0, 6), pady=4)

            # Right: source viewer
            viewer = CodexBox(tab, font=ctk.CTkFont(family=FONT_MONO[0], size=10))
            viewer.grid(row=0, column=1, sticky="nsew", pady=4)
            viewer.set_text(f"  {G.DOT_DIM}  pick a file from the left.")

            for path, purpose in items:
                p = ROOT / path
                exists = p.exists()
                glyph = G.DOT_ON if exists else G.DOT_OFF
                btn = ctk.CTkButton(
                    left,
                    text=f"  {glyph}  {path.split('/')[-1]}",
                    anchor="w",
                    fg_color="transparent",
                    hover_color=C.PANEL_HI,
                    text_color=C.BONE if exists else C.GHOST,
                    font=ctk.CTkFont(family=FONT_MONO[0], size=11),
                    height=24, corner_radius=2,
                    command=(lambda pth=p, pur=purpose, v=viewer:
                              self._show_file(pth, pur, v)),
                )
                btn.pack(fill="x", padx=4, pady=(2, 0))
                caption = ctk.CTkLabel(
                    left, text=f"        {purpose}",
                    text_color=C.ASH, anchor="w",
                    font=ctk.CTkFont(family=FONT_SANS[0], size=9),
                    justify="left", wraplength=270,
                )
                caption.pack(fill="x", padx=4, pady=(0, 4))

        self._build_vs_explorer_tab()

    # ------------------------------------------------------------------
    def _build_vs_explorer_tab(self) -> None:
        """VS Code-style file explorer showing all project files."""
        tab = self.tabview.add("VS EXPLORER")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Outer frame fills the tab
        outer = ctk.CTkFrame(tab, fg_color=C.NIGHT, corner_radius=2)
        outer.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(0, weight=1)

        # ttk style for dark treeview
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "VSExplorer.Treeview",
            background=C.OBSIDIAN,
            foreground=C.BONE,
            fieldbackground=C.OBSIDIAN,
            font=("Courier New", 9),
            rowheight=16,
            borderwidth=0,
        )
        style.configure(
            "VSExplorer.Treeview.Heading",
            background=C.PANEL_HI,
            foreground=C.ASH,
            font=("Courier New", 8, "bold"),
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "VSExplorer.Treeview",
            background=[("selected", C.PANEL_HI)],
            foreground=[("selected", C.BONE)],
        )

        # Scrollbar + treeview container
        tree_frame = ctk.CTkFrame(outer, fg_color=C.OBSIDIAN, corner_radius=0)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree = ttk.Treeview(
            tree_frame,
            columns=("lines", "size", "modified"),
            show="tree headings",
            style="VSExplorer.Treeview",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            selectmode="browse",
        )
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.configure(command=tree.yview)
        hsb.configure(command=tree.xview)

        # Column widths / headings
        tree.column("#0",      width=260, minwidth=120, stretch=True)
        tree.column("lines",   width=55,  minwidth=40,  stretch=False, anchor="e")
        tree.column("size",    width=60,  minwidth=50,  stretch=False, anchor="e")
        tree.column("modified",width=85,  minwidth=70,  stretch=False, anchor="e")
        tree.heading("#0",       text="name",     anchor="w")
        tree.heading("lines",    text="lines",    anchor="e")
        tree.heading("size",     text="size",     anchor="e")
        tree.heading("modified", text="modified", anchor="e")

        # Colour tags
        tree.tag_configure("dir",      foreground=C.BONE,  font=("Courier New", 9, "bold"))
        tree.tag_configure("py",       foreground=C.LIFE)
        tree.tag_configure("json",     foreground=C.OMEN)
        tree.tag_configure("yaml",     foreground=C.WIND)
        tree.tag_configure("shell",    foreground=C.WOUND)
        tree.tag_configure("other",    foreground=C.ASH)

        # ---- helpers ----
        SKIP_DIRS  = {".git", "__pycache__", ".pytest_cache", "venv",
                      "node_modules", ".DS_Store"}
        SKIP_EXTS  = {".pyc", ".pyo", ".pkl", ".h5", ".pt",
                      ".bin", ".so", ".dylib"}

        def _icon(suffix: str) -> str:
            return {"py": "🐍 ", "json": "📋 ", "yaml": "⚙ ", "yml": "⚙ ",
                    "sh": "$ ", "command": "$ ", "md": "📄 ", "txt": "≡ "}.get(
                suffix.lstrip("."), "· ")

        def _tag(suffix: str) -> str:
            s = suffix.lstrip(".")
            if s == "py":                return "py"
            if s == "json":              return "json"
            if s in ("yaml", "yml"):     return "yaml"
            if s in ("sh", "command"):   return "shell"
            return "other"

        def _human(n: int) -> str:
            return f"{n/1024:.1f}K" if n >= 1024 else f"{n}B"

        def _lines(p: Path) -> str:
            try:
                return str(p.read_text(errors="replace").count("\n") + 1)
            except Exception:
                return "—"

        # ---- recursive builder ----
        total_files: list = []
        total_lines_acc: list = []

        def _insert_dir(parent_iid: str, dir_path: Path, depth: int) -> None:
            try:
                entries = sorted(dir_path.iterdir(),
                                 key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                return

            for entry in entries:
                if entry.name in SKIP_DIRS or entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    iid = tree.insert(
                        parent_iid, "end",
                        text=f"  {entry.name}/",
                        values=("", "", ""),
                        tags=("dir",),
                        open=(depth < 2),
                    )
                    _insert_dir(iid, entry, depth + 1)
                elif entry.is_file():
                    suffix = entry.suffix.lower()
                    if suffix in SKIP_EXTS:
                        continue
                    try:
                        stat   = entry.stat()
                        mtime  = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
                        sz     = _human(stat.st_size)
                    except Exception:
                        mtime, sz = "—", "—"
                    lc = _lines(entry)
                    icon = _icon(suffix)
                    tag  = _tag(suffix)
                    tree.insert(
                        parent_iid, "end",
                        text=f"  {icon}{entry.name}",
                        values=(lc, sz, mtime),
                        tags=(tag,),
                    )
                    total_files.append(1)
                    try:
                        total_lines_acc.append(int(lc))
                    except (ValueError, TypeError):
                        pass

        _insert_dir("", ROOT, 0)

        # ---- status bar ----
        n_files = sum(total_files)
        n_lines = sum(total_lines_acc)
        self._vs_status_label = ctk.CTkLabel(
            outer,
            text=f"  {n_files} files  ·  {n_lines:,} lines",
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_MONO[0], size=9),
            anchor="w",
        )
        self._vs_status_label.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 0))

        # ---- refresh button ----
        def _refresh() -> None:
            for iid in tree.get_children():
                tree.delete(iid)
            total_files.clear()
            total_lines_acc.clear()
            _insert_dir("", ROOT, 0)
            n = sum(total_files); l = sum(total_lines_acc)
            self._vs_status_label.configure(
                text=f"  {n} files  ·  {l:,} lines")

        refresh_btn = ctk.CTkButton(
            outer,
            text="↻  Refresh",
            width=80, height=20,
            fg_color=C.PANEL_HI,
            hover_color=C.PANEL,
            text_color=C.ASH,
            font=ctk.CTkFont(family=FONT_MONO[0], size=9),
            corner_radius=2,
            command=_refresh,
        )
        refresh_btn.grid(row=2, column=0, sticky="e", padx=6, pady=(2, 4))

    def _show_file(self, path: Path, purpose: str, viewer: CodexBox) -> None:
        try:
            if not path.exists():
                viewer.set_text(f"  {G.SKULL}  file not found: {path}")
                return
            text = path.read_text(errors="replace")
            if len(text) > 200_000:
                text = text[:200_000] + f"\n\n... [truncated at 200 KB] ..."
            header = (
                f"  {G.CROSS}  {path.relative_to(ROOT)}\n"
                f"  ────────────────────────────────────────────────────────────\n"
                f"  purpose: {purpose}\n"
                f"  size:    {path.stat().st_size:,} bytes\n"
                f"  ────────────────────────────────────────────────────────────\n\n"
            )
            viewer.set_text(header + text)
        except Exception as exc:  # noqa: BLE001
            viewer.set_text(f"  {G.SKULL}  read failed: {exc}")

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
        n_src = sum(1 for _ in SRC_DIR.rglob("*.py")) if SRC_DIR.exists() else 0
        n_data = sum(1 for _ in DATA_DIR.rglob("*")) if DATA_DIR.exists() else 0
        n_models = sum(1 for _ in MODELS_DIR.glob("*")) if MODELS_DIR.exists() else 0
        total_mapped = sum(len(v) for v in CODE_CATEGORIES.values())
        self.m_files.set(str(n_src), sub=f"{total_mapped} in codex")
        self.m_data.set(str(n_data))
        self.m_models.set(str(n_models),
                           sub=("trained" if (MODELS_DIR / "xgb.json").exists()
                                else "heuristic"))

        log_file = ROOT / "trading_enhancer.log"
        if log_file.exists():
            mt = log_file.stat().st_mtime
            if mt > self._last_log_mtime:
                self.bar_log.pulse(0.9); self._last_log_mtime = mt
            else: self.bar_log.pulse(0.05)
        else: self.bar_log.pulse(0.02)

        sig_dir = DATA_DIR / "signals"
        n_sig = sum(1 for _ in sig_dir.glob("*.json")) if sig_dir.exists() else 0
        if n_sig > self._last_signal_count:
            self.bar_signals.pulse(0.95); self._last_signal_count = n_sig
        else: self.bar_signals.pulse(0.04)

        n_ord = len(snap.get("orders", []))
        if n_ord != self._last_order_count:
            self.bar_orders.pulse(0.8); self._last_order_count = n_ord
        else: self.bar_orders.pulse(0.05)

        pm_file = DATA_DIR / "postmortems.jsonl"
        if pm_file.exists():
            try:
                cur = sum(1 for _ in pm_file.read_text().splitlines() if _.strip())
            except Exception:
                cur = 0
            if cur != self._last_lesson_count:
                self.bar_lessons.pulse(0.85); self._last_lesson_count = cur
            else: self.bar_lessons.pulse(0.06)
        else: self.bar_lessons.pulse(0.02)

    # ------------------------------------------------------------------
    def _inscribe(self) -> None:
        out_dir = DATA_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = out_dir / f"system_report_{ts}.md"
        try:
            track = ROOT / "data" / "track_record.jsonl"
            n_sessions = (sum(1 for _ in track.read_text().splitlines() if _.strip())
                           if track.exists() else 0)
            pm = ROOT / "data" / "postmortems.jsonl"
            n_lessons = (sum(1 for _ in pm.read_text().splitlines() if _.strip())
                          if pm.exists() else 0)
            sig_dir = ROOT / "data" / "signals"
            n_signals = (sum(1 for _ in sig_dir.glob("*.json"))
                          if sig_dir.exists() else 0)
            models = list((ROOT / "models").glob("*")) if (ROOT / "models").exists() else []
            content = (
                f"# BRZRKR System Report — {ts}\n\n## Counts\n\n"
                f"- Sessions logged:        {n_sessions}\n"
                f"- Lessons in postmortem:  {n_lessons}\n"
                f"- Signals on disk:        {n_signals}\n"
                f"- Model artifacts:        {len(models)}\n\n## Source files by category\n"
            )
            for cat, items in CODE_CATEGORIES.items():
                content += f"\n### {cat}\n\n"
                for path, desc in items:
                    content += f"- `{path}` — {desc}\n"
            out.write_text(content)
            self.exp_status.configure(text=f"{G.DOT_ON}  Inscribed: {out}",
                                       text_color=C.LIFE)
        except Exception as exc:  # noqa: BLE001
            self.exp_status.configure(text=f"{G.SKULL}  Failed: {exc}",
                                       text_color=C.DEATH)

    def _inscribe_map(self) -> None:
        out_dir = DATA_DIR / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "project_tree.txt"
        skip = {"venv", "__pycache__", ".git", "build", "dist",
                ".pytest_cache", "_legacy"}
        lines = []
        def walk(p: Path, prefix: str = "") -> None:
            try:
                items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            except Exception:
                return
            items = [i for i in items if i.name not in skip
                      and not i.name.startswith(".")]
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}{item.name}")
                if item.is_dir():
                    walk(item, prefix + ("    " if is_last else "│   "))
        lines.append(f"BRZRKR/ (rooted at {ROOT.name})")
        walk(ROOT)
        out.write_text("\n".join(lines))
        self.exp_status.configure(text=f"{G.DOT_ON}  Map inscribed: {out}",
                                   text_color=C.LIFE)

    def _inscribe_postmortem(self) -> None:
        try:
            from src.learning.postmortem_db import PostmortemDB
            db = PostmortemDB()
            db.bootstrap_if_empty()
            md = db.to_markdown(top=100)
            out = ROOT / "docs" / "postmortems.md"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md)
            self.exp_status.configure(text=f"{G.DOT_ON}  Codex inscribed: {out}",
                                       text_color=C.LIFE)
        except Exception as exc:  # noqa: BLE001
            self.exp_status.configure(text=f"{G.SKULL}  Failed: {exc}",
                                       text_color=C.DEATH)
