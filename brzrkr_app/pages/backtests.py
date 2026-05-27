"""Backtests page — past test success rates + 2x2 candlestick grid.

Reads results from:
  data/scenario_runs/_results_so_far.csv     (weekend_practice.py output)
  data/scenario_runs/scenario_report_*.csv   (older reports)
  data/backtest_out/equity_curve.csv         (single-run backtest)

Shows:
  - Top metrics: total scenarios run, win rate, mean return vs benchmark
  - 2×2 grid of mini candlestick charts (the 4 most recent / interesting)
  - Per-category breakdown table
  - Scrollable list of all runs with a click-to-load detail
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import pandas as pd

from brzrkr_app.theme import (
    C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, pnl_color,
)
from brzrkr_app.widgets import (
    BarMeter, BloodMetric, CodexBox, EquityCurve, GothicCard, InkDivider,
    LiveSimCard, MiniCandleChart, PageTitle, RuneButton,
    SectionHeader, StatusBeacon,
)

ROOT = Path(__file__).resolve().parent.parent.parent
SCENARIO_DIR = ROOT / "data" / "scenario_runs"
BACKTEST_DIR = ROOT / "data" / "backtest_out"


class BacktestsPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app

        # Scrollable root so content > screen.
        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C.NIGHT)
        self._scroll.pack(fill="both", expand=True)
        body = self._scroll
        body.grid_columnconfigure((0, 1, 2, 3), weight=1)

        PageTitle(body, f"{G.RUNE_O} Proving Grounds",
                   subtitle="past tests · scenario battery · proof of strategy"
                   ).grid(row=0, column=0, columnspan=4, sticky="ew",
                          pady=(0, 4))
        InkDivider(body, length=720).grid(row=1, column=0, columnspan=4,
                                            sticky="w", pady=(0, 12))

        # ---- AUTOMATION (continuous practice loop) ------------------
        auto_card = GothicCard(body)
        auto_card.grid(row=2, column=0, columnspan=4, sticky="ew",
                        pady=(0, 12))
        auto_card.grid_columnconfigure(0, weight=1)
        SectionHeader(auto_card, "Continuous Practice",
                       glyph=G.RUNE_O).grid(row=0, column=0, sticky="ew")
        ctrl = ctk.CTkFrame(auto_card, fg_color="transparent")
        ctrl.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        ctrl.grid_columnconfigure(0, weight=1)
        self.cont_status = StatusBeacon(ctrl, "—", "neutral")
        self.cont_status.grid(row=0, column=0, sticky="w")
        RuneButton(ctrl, "Start Continuous", glyph=G.EXEC,
                    command=self._start_continuous
                    ).grid(row=0, column=1, padx=(8, 4))
        from brzrkr_app.widgets import GhostButton
        GhostButton(ctrl, "Stop", glyph=G.SHIELD,
                     command=self._stop_continuous
                     ).grid(row=0, column=2)

        # ---- LIVE SIMULATIONS — 2×2 slot grid -----------------------
        live_card = GothicCard(body)
        live_card.grid(row=3, column=0, columnspan=4, sticky="ew",
                        pady=(0, 12))
        live_card.grid_columnconfigure(0, weight=1)

        SectionHeader(live_card, "Live Simulations (4 slots)",
                       glyph=G.RUNE_T).grid(row=0, column=0, sticky="ew")

        # Header line: beacon + overall progress
        live_head = ctk.CTkFrame(live_card, fg_color="transparent")
        live_head.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        live_head.grid_columnconfigure(1, weight=1)
        self.live_beacon = StatusBeacon(live_head, "checking…", "neutral")
        self.live_beacon.grid(row=0, column=0, sticky="w")
        self.bar_overall = BarMeter(live_head, label="overall",
                                      width=380, max_value=100, unit="%",
                                      color=C.SIGIL)
        self.bar_overall.grid(row=0, column=1, sticky="e", padx=(20, 0))

        # 2×2 grid of LiveSimCards
        grid = ctk.CTkFrame(live_card, fg_color="transparent")
        grid.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 14))
        grid.grid_columnconfigure((0, 1), weight=1)
        grid.grid_rowconfigure((0, 1), weight=1)
        self.slot_cards: list[LiveSimCard] = []
        for i in range(4):
            r, ccol = divmod(i, 2)
            card = LiveSimCard(grid, slot=i)
            card.grid(row=r, column=ccol, padx=4, pady=4, sticky="nsew")
            self.slot_cards.append(card)

        # ---- Existing top metric strip --------------------------------
        self.m_runs   = BloodMetric(body, "Total Runs", "0")
        self.m_winrate = BloodMetric(body, "Mean Win Rate", "—")
        self.m_strat  = BloodMetric(body, "Strategy Return", "—")
        self.m_bench  = BloodMetric(body, "Vs Benchmark", "—")
        self.m_runs.grid(row=4, column=0, padx=(0, 6), sticky="nsew",
                          pady=(12, 0))
        self.m_winrate.grid(row=4, column=1, padx=6, sticky="nsew",
                             pady=(12, 0))
        self.m_strat.grid(row=4, column=2, padx=6, sticky="nsew",
                           pady=(12, 0))
        self.m_bench.grid(row=4, column=3, padx=(6, 0), sticky="nsew",
                           pady=(12, 0))

        # No more mini candle grid — live cards above cover that role.
        self.charts: list = []

        # By-category breakdown
        cat_card = GothicCard(body)
        cat_card.grid(row=6, column=0, columnspan=2, sticky="nsew",
                        padx=(0, 6), pady=(12, 0))
        cat_card.grid_columnconfigure(0, weight=1)
        cat_card.grid_rowconfigure(1, weight=1)
        SectionHeader(cat_card, "By Category", glyph=G.CROSS).grid(
            row=0, column=0, sticky="ew")
        self.cat_box = CodexBox(cat_card, height=200)
        self.cat_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

        # All runs list
        runs_card = GothicCard(body)
        runs_card.grid(row=6, column=2, columnspan=2, sticky="nsew",
                        padx=(6, 0), pady=(12, 0))
        runs_card.grid_columnconfigure(0, weight=1)
        runs_card.grid_rowconfigure(1, weight=1)
        SectionHeader(runs_card, "All Runs", glyph=G.RUNE_O).grid(
            row=0, column=0, sticky="ew")
        self.runs_box = CodexBox(runs_card, height=200)
        self.runs_box.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="nsew")

        # ---- SIM BREAKDOWN (day / week / month) ----------------------
        sim_card = GothicCard(body)
        sim_card.grid(row=7, column=0, columnspan=4, sticky="ew",
                       pady=(12, 0))
        sim_card.grid_columnconfigure((0, 1, 2, 3), weight=1)
        SectionHeader(sim_card, "Simulation Results by Period — $100k Start",
                       glyph=G.RUNE_T).grid(row=0, column=0, columnspan=4,
                                              sticky="ew")

        # Period metric triplet inside a sub-grid
        self._sim_metrics: dict = {}
        period_defs = [
            ("1-Day",   "day",   "Intraday scalp / swing"),
            ("1-Week",  "week",  "Multi-day momentum"),
            ("1-Month", "month", "Trend following"),
            ("Full",    "full",  "All combined runs"),
        ]
        sim_grid = ctk.CTkFrame(sim_card, fg_color="transparent")
        sim_grid.grid(row=1, column=0, columnspan=4, sticky="ew",
                       padx=14, pady=(0, 14))
        for i in range(4):
            sim_grid.grid_columnconfigure(i, weight=1)

        for col, (label, key, sub) in enumerate(period_defs):
            pcard = GothicCard(sim_grid)
            pcard.grid(row=0, column=col,
                        padx=(0, 6) if col < 3 else 0, sticky="nsew")
            pcard.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                pcard, text=label,
                font=ctk.CTkFont(family=FONT_DISPLAY[0], size=13, weight="bold"),
                text_color=C.SIGIL, anchor="w",
            ).grid(row=0, column=0, padx=14, pady=(10, 0), sticky="ew")
            ctk.CTkLabel(pcard, text=sub, text_color=C.GHOST,
                          font=ctk.CTkFont(family=FONT_SANS[0], size=8),
                          anchor="w",
                          ).grid(row=1, column=0, padx=14, pady=(0, 4), sticky="ew")
            m_eq   = BloodMetric(pcard, "Equity",   "——")
            m_ret  = BloodMetric(pcard, "Return",   "——")
            m_wr   = BloodMetric(pcard, "Win Rate", "——")
            m_eq.grid( row=2, column=0, padx=10, pady=(0, 1), sticky="ew")
            m_ret.grid(row=3, column=0, padx=10, pady=1,       sticky="ew")
            m_wr.grid( row=4, column=0, padx=10, pady=(1, 10), sticky="ew")
            self._sim_metrics[key] = (m_eq, m_ret, m_wr)

        # ---- LEARNING PROGRESSION ------------------------------------
        learn_card = GothicCard(body)
        learn_card.grid(row=8, column=0, columnspan=4, sticky="ew",
                          pady=(12, 0))
        learn_card.grid_columnconfigure(0, weight=1)
        learn_card.grid_columnconfigure(1, weight=1)
        SectionHeader(learn_card, "Self-Teaching — Learning Progression",
                       glyph=G.SKULL).grid(row=0, column=0, columnspan=2,
                                            sticky="ew")

        lrow = ctk.CTkFrame(learn_card, fg_color="transparent")
        lrow.grid(row=1, column=0, columnspan=2, sticky="ew",
                   padx=14, pady=(0, 14))
        lrow.grid_columnconfigure(0, weight=1)
        lrow.grid_columnconfigure(1, weight=1)

        self._learn_progress_box = CodexBox(lrow, height=130)
        self._learn_progress_box.grid(row=0, column=0, padx=(0, 6),
                                        sticky="nsew")
        self._learn_history_box = CodexBox(lrow, height=130)
        self._learn_history_box.grid(row=0, column=1, padx=(6, 0),
                                       sticky="nsew")

        # Footer: where data comes from
        foot = GothicCard(body)
        foot.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        foot.grid_columnconfigure(0, weight=1)
        SectionHeader(foot, "Proof of Strategy — Sources", glyph=G.DAGGER
                       ).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            foot,
            text=(f"  {G.RIGHT}  Run `python weekend_practice.py` to populate this tab.\n"
                  f"  {G.RIGHT}  Single backtests: `python run.py --mode backtest --data data/sample.csv`\n"
                  f"  {G.RIGHT}  Results written to: data/scenario_runs/  and  data/backtest_out/\n"
                  f"  {G.RIGHT}  See CSVs + markdown reports in those folders for the full ledger."),
            justify="left", anchor="w",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_SERIF[0], size=11),
        ).grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        # Background load — deferred so mainloop is running when thread fires
        self.after(500, self._reload_in_thread)
        self.after(600, self._refresh_learning_progression)

        # Live status polling (separate, faster loop than broker poll)
        self.after(800, self._poll_live)

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
        # Backtest data refreshes from disk, not from broker snapshots.
        # We refresh once per N broker ticks to avoid hammering disk.
        if not hasattr(self, "_ticks_since_reload"):
            self._ticks_since_reload = 0
        self._ticks_since_reload += 1
        if self._ticks_since_reload >= 4:  # every ~32s
            self._ticks_since_reload = 0
            self._reload_in_thread()

    def _reload_in_thread(self) -> None:
        threading.Thread(target=self._reload, daemon=True).start()

    def _reload(self) -> None:
        try:
            df = self._load_scenarios()
        except Exception:
            df = pd.DataFrame()
        # Push UI updates back to the main thread safely.
        try:
            self.after(0, self._apply, df)
        except RuntimeError:
            pass  # widget destroyed or not in main loop

    def _load_scenarios(self) -> pd.DataFrame:
        # Prefer the partial (most recent) file, fall back to dated reports.
        partial = SCENARIO_DIR / "_results_so_far.csv"
        if partial.exists():
            return pd.read_csv(partial)
        # Latest dated report
        reports = sorted(SCENARIO_DIR.glob("scenario_report_*.csv"))
        if reports:
            return pd.read_csv(reports[-1])
        return pd.DataFrame()

    def _apply(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.m_runs.set("0", sub="run weekend_practice.py")
            self.m_winrate.set("—")
            self.m_strat.set("—")
            self.m_bench.set("—")
            self.cat_box.set_text(f"  {G.DOT_DIM}  no scenario data yet.\n\n"
                                    "  Run: python weekend_practice.py")
            self.runs_box.set_text(f"  {G.DOT_DIM}  no runs to display.")
            for ch in self.charts:
                ch.set_data(None, title="—")
            return

        clean = df[~df.get("failed", False).fillna(False).astype(bool)] \
            if "failed" in df.columns else df

        # Metrics
        n = len(df)
        if not clean.empty:
            mean_wr = clean["win_rate"].mean() * 100
            self.m_winrate.set(f"{mean_wr:.1f}%",
                                color=C.LIFE if mean_wr >= 50 else C.WOUND)
            strat = ((clean["final_equity"] - 100_000.0) / 100_000.0 * 100).mean()
            self.m_strat.set(f"{strat:+.2f}%",
                              color=pnl_color(strat))
            if "relative_vs_benchmark_pct" in clean.columns:
                bench = clean["relative_vs_benchmark_pct"].mean()
                self.m_bench.set(f"{bench:+.2f}%",
                                  sub="vs buy-and-hold",
                                  color=pnl_color(bench))
            else:
                self.m_bench.set("—")
        self.m_runs.set(str(n),
                          sub=(f"{int(df['failed'].sum())} failed"
                                if "failed" in df.columns
                                else ""))

        # By-category breakdown
        if "category" in clean.columns and not clean.empty:
            lines = []
            for cat, sub in clean.groupby("category"):
                wr = sub["win_rate"].mean() * 100
                sret = ((sub["final_equity"] - 100_000) / 100_000 * 100).mean()
                trades = sub["trades"].mean()
                lines.append(
                    f"  {G.RUNE_T}  {cat.upper():<14}  runs={len(sub):>2}  "
                    f"win {wr:>5.1f}%  ret {sret:>+6.2f}%  trades {trades:>4.1f}"
                )
            self.cat_box.set_text("\n".join(lines) if lines
                                    else f"  {G.DOT_DIM}  no clean data")
        else:
            self.cat_box.set_text(f"  {G.DOT_DIM}  no category column in data.")

        # All runs list
        lines = []
        for _, row in df.iterrows():
            failed = bool(row.get("failed", False))
            sym = G.SKULL if failed else G.DOT_ON
            sret = (row.get("final_equity", 100_000) - 100_000) / 100_000 * 100
            lines.append(
                f"  {sym}  {str(row.get('scenario','?'))[:25]:<25}  "
                f"{str(row.get('symbol','?')):<5}  "
                f"trades={int(row.get('trades', 0)):>3}  "
                f"ret={sret:>+6.2f}%"
            )
        self.runs_box.set_text("\n".join(lines))

        # (no more per-row chart loading — replaced by live slot cards above)

        # Refresh sim breakdown + learning progression
        self._apply_sim_breakdown(df)
        self._refresh_learning_progression()

    def _apply_sim_breakdown(self, df: pd.DataFrame) -> None:
        """Populate the day/week/month/full simulation metric cards."""
        if not hasattr(self, "_sim_metrics"):
            return

        # Map period keys to scenario category filters
        # The scenario CSV may have a 'period', 'category', or 'scenario' col
        def _slice(key: str) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame()
            failed_mask = df.get("failed", pd.Series(False, index=df.index))
            clean = df[~failed_mask.fillna(False).astype(bool)]
            if clean.empty:
                return clean
            if key == "full":
                return clean
            # Try matching on scenario name or category
            for col in ("period", "category", "scenario"):
                if col in clean.columns:
                    sub = clean[clean[col].astype(str).str.lower().str.contains(
                        key, na=False)]
                    if not sub.empty:
                        return sub
            return pd.DataFrame()

        for key, (m_eq, m_ret, m_wr) in self._sim_metrics.items():
            sub = _slice(key)
            if sub.empty:
                m_eq.set("——")
                m_ret.set("——")
                m_wr.set("——")
                continue
            eq_col  = "final_equity" if "final_equity" in sub.columns else None
            wr_col  = "win_rate"     if "win_rate"     in sub.columns else None
            if eq_col:
                mean_eq  = sub[eq_col].mean()
                mean_ret = (mean_eq - 100_000) / 100_000 * 100
                m_eq.set(f"${mean_eq:,.0f}", color=pnl_color(mean_ret))
                m_ret.set(f"{mean_ret:+.2f}%", color=pnl_color(mean_ret))
            else:
                m_eq.set("——"); m_ret.set("——")
            if wr_col:
                wr = sub[wr_col].mean() * 100
                m_wr.set(f"{wr:.1f}%", color=C.LIFE if wr >= 50 else C.WOUND)
            else:
                m_wr.set("——")

    def _refresh_learning_progression(self) -> None:
        """Populate the learning progression boxes from learning reports."""
        if not hasattr(self, "_learn_progress_box"):
            return

        # ── Left box: current model state ────────────────────────────
        prog_lines = []
        try:
            lr = None
            lr_path = Path("data/learning_report.json")
            if lr_path.exists():
                import json
                lr = json.loads(lr_path.read_text())
            if lr:
                prog_lines.append(f"  {G.CROSS}  CURRENT MODEL")
                prog_lines.append(f"  Retrained:  {lr.get('retrained_at','—')[:16]}")
                prog_lines.append(f"  Rows used:  {lr.get('training_rows',0)}")
                prog_lines.append(f"  In-sample accuracy:  {lr.get('in_sample_acc',0)*100:.1f}%")
                prog_lines.append(f"  Win rate (closed):   {lr.get('win_rate',0)*100:.1f}%")
                prog_lines.append(f"  Avg R-multiple:      {lr.get('avg_r',0):.3f}")
                prog_lines.append(f"  Confidence thr:      {lr.get('suggested_confidence_threshold',0.75):.3f}")
                prog_lines.append("")
                prog_lines.append(f"  {G.CROSS}  ONLINE LEARNER")
                prog_lines.append(f"  Trees appended each refit: +50")
                prog_lines.append(f"  Min new trades to refit: 5")
                prog_lines.append(f"  Features: 14-dim ensemble input")
                prog_lines.append(f"  Model: XGBoost (incremental) + LSTM + Transformer")
        except Exception as exc:
            prog_lines.append(f"  No learning data yet.")
            prog_lines.append(f"  Run the agent to generate trades.")

        try:
            from src.learning.trade_journal import TradeJournal
            s = TradeJournal().stats()
            prog_lines.append("")
            prog_lines.append(f"  {G.CROSS}  JOURNAL")
            prog_lines.append(f"  Closed:  {s['total_closed']}  Open: {s['open']}")
            prog_lines.append(f"  Wins: {s['wins']}  Losses: {s['losses']}")
            prog_lines.append(f"  Win rate: {s['win_rate']*100:.1f}%")
            prog_lines.append(f"  Total P&L: ${s['total_pnl_usd']:+,.2f}")
        except Exception:
            pass

        self._learn_progress_box.set_text("\n".join(prog_lines) if prog_lines
                                            else "  No learning data yet.")

        # ── Right box: historical accuracy by iteration ───────────────
        hist_lines = []
        try:
            state_path = Path("data/learner_state.json")
            if state_path.exists():
                import json
                state = json.loads(state_path.read_text())
                hist  = state.get("accuracy_history", [])
                if hist:
                    hist_lines.append(f"  {G.CROSS}  ACCURACY HISTORY  ({len(hist)} updates)")
                    hist_lines.append(f"  {'Iter':<5}  {'Acc':>6}  {'WinR':>6}  {'Rows':>6}")
                    hist_lines.append(f"  {'─'*32}")
                    for i, h in enumerate(hist[-12:], 1):   # last 12 updates
                        acc  = h.get("in_sample_acc", 0) * 100
                        wr   = h.get("win_rate", 0) * 100
                        rows = h.get("training_rows", 0)
                        hist_lines.append(
                            f"  {i:<5}  {acc:>5.1f}%  {wr:>5.1f}%  {rows:>6}")
                else:
                    hist_lines.append("  No training iterations yet.")
            else:
                hist_lines.append(f"  {G.CROSS}  TEACHING LOOP")
                hist_lines.append("  Triggers automatically after 5+ closed trades.")
                hist_lines.append("  Each refit appends 50 XGBoost trees on new data.")
                hist_lines.append("  Regime detector recalibrates thresholds hourly.")
                hist_lines.append("")
                hist_lines.append("  WHAT IT LEARNS:")
                hist_lines.append("  · Which features predicted wins vs losses")
                hist_lines.append("  · When to stay out (low confidence setups)")
                hist_lines.append("  · Session/regime performance differentials")
                hist_lines.append("  · Volume anomaly → success rate correlation")
        except Exception:
            hist_lines.append("  Learning state file not found.")
            hist_lines.append("  Will populate after first agent run.")

        self._learn_history_box.set_text("\n".join(hist_lines) if hist_lines
                                           else "  No history yet.")

    # ------------------------------------------------------------------
    # Continuous-mode controls
    # ------------------------------------------------------------------
    _CONT_PID_FILE = ROOT / ".continuous.pid"
    _CONT_STOP_FILE = ROOT / "CONTINUOUS_STOP"
    _CONT_LOG = ROOT / "continuous.out"

    def _continuous_running(self) -> tuple[bool, int]:
        if not self._CONT_PID_FILE.exists():
            return (False, 0)
        try:
            pid = int(self._CONT_PID_FILE.read_text().strip())
            import os
            os.kill(pid, 0)
            return (True, pid)
        except Exception:
            return (False, 0)

    def _start_continuous(self) -> None:
        running, _ = self._continuous_running()
        if running:
            self.app.toast("Continuous practice is already running.")
            return
        if self._CONT_STOP_FILE.exists():
            try: self._CONT_STOP_FILE.unlink()
            except Exception: pass
        import subprocess, sys
        cmd = [sys.executable, str(ROOT / "continuous_practice.py"),
               "--parallel", "4", "--rest", "30"]
        log = open(self._CONT_LOG, "ab")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(ROOT))
        self._CONT_PID_FILE.write_text(str(proc.pid))
        self.app.toast(f"{G.RUNE_T}  Continuous practice started (PID {proc.pid}).")

    def _stop_continuous(self) -> None:
        running, pid = self._continuous_running()
        if not running:
            self.app.toast("Continuous practice isn't running.")
            return
        self._CONT_STOP_FILE.touch()
        self.app.toast("Stop sigil written — finishing current batch.")

    # ------------------------------------------------------------------
    # Live status polling
    # ------------------------------------------------------------------
    def _poll_live(self) -> None:
        try:
            self._refresh_live()
            self._refresh_continuous_status()
        except Exception:
            pass
        self.after(400, self._poll_live)   # ~2.5 Hz

    def _refresh_continuous_status(self) -> None:
        running, pid = self._continuous_running()
        if running:
            self.cont_status.set(
                f"Running  ·  PID {pid}  ·  log: continuous.out",
                "ok",
            )
        elif self._CONT_STOP_FILE.exists():
            self.cont_status.set("Stopping at end of current batch", "warn")
        else:
            self.cont_status.set("Not running", "neutral", glyph=G.DOT_DIM)

    def _refresh_live(self) -> None:
        from src.backtest.live_status import LiveStatusWriter
        # Slot 0 lives at _live.json; slots 1+ at _live_<i>.json.
        slot_paths = [
            SCENARIO_DIR / "_live.json",
            SCENARIO_DIR / "_live_1.json",
            SCENARIO_DIR / "_live_2.json",
            SCENARIO_DIR / "_live_3.json",
        ]
        statuses = [LiveStatusWriter.read(p) for p in slot_paths]
        any_active = any(s and s.get("active") for s in statuses)
        any_present = any(s is not None for s in statuses)

        # Overall progress: any slot with the overall_done/total fields.
        done = total = 0
        for s in statuses:
            if s and s.get("scenarios_total"):
                done = max(done, int(s.get("scenarios_done", 0)))
                total = max(total, int(s.get("scenarios_total", 0)))

        if any_active:
            n_active = sum(1 for s in statuses if s and s.get("active"))
            self.live_beacon.set(
                f"{n_active} sim(s) running   ·   {done}/{total} scenarios done",
                "ok",
            )
        elif any_present:
            self.live_beacon.set(
                f"battery idle ({done}/{total})", "neutral", glyph=G.DOT_DIM)
        else:
            self.live_beacon.set(
                "no simulations tracked yet", "neutral", glyph=G.DOT_DIM)
        self.bar_overall.set((done / total * 100) if total else 0)

        # Push each slot's state into its card
        for card, status in zip(self.slot_cards, statuses):
            card.update_from(status)


# Late import to avoid circular imports at module load time.
from brzrkr_app.theme import FONT_SERIF  # noqa: E402
