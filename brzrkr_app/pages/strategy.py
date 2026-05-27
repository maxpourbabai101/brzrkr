"""Strategy page — proof of improvement over time.

Shows:
  - Equity over sessions (line chart from track_record.jsonl)
  - Promotion gate readiness (progress bar against gate criteria)
  - Trained model status + last training report
  - Lessons learned over time (counter + breakdown)
  - Last correlation analysis (which lessons matter most)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import customtkinter as ctk

from brzrkr_app.theme import C, G, FONT_DISPLAY, FONT_MONO, FONT_SANS, FONT_SERIF, pnl_color
from brzrkr_app.widgets import (
    BloodMetric, CodexBox, EquityCurve, GothicCard, InkDivider,
    PageTitle, RuneButton, SectionHeader, StatusBeacon,
)

ROOT = Path(__file__).resolve().parent.parent.parent
TRACK_FILE = ROOT / "data" / "track_record.jsonl"
POSTMORTEM_FILE = ROOT / "data" / "postmortems.jsonl"
MODEL_FILE = ROOT / "models" / "xgb.json"
MODEL_REPORT = ROOT / "models" / "xgb_report.json"


class StrategyPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=C.NIGHT)
        self.app = app

        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C.NIGHT)
        self._scroll.pack(fill="both", expand=True)
        body = self._scroll
        body.grid_columnconfigure((0, 1, 2, 3), weight=1)

        PageTitle(body, f"{G.RUNE_T} The Forging",
                   subtitle="proof of progress · sharpening of strategy"
                   ).grid(row=0, column=0, columnspan=4, sticky="ew",
                          pady=(0, 4))
        InkDivider(body, length=720).grid(row=1, column=0, columnspan=4,
                                            sticky="w", pady=(0, 12))

        # Top metrics
        self.m_sessions = BloodMetric(body, "Sessions Logged", "0")
        self.m_pnl = BloodMetric(body, "Cumulative P&L", "—")
        self.m_winrate = BloodMetric(body, "Session Win Rate", "—")
        self.m_lessons = BloodMetric(body, "Lessons Forged", "0")
        self.m_sessions.grid(row=2, column=0, padx=(0, 6), sticky="nsew")
        self.m_pnl.grid(row=2, column=1, padx=6, sticky="nsew")
        self.m_winrate.grid(row=2, column=2, padx=6, sticky="nsew")
        self.m_lessons.grid(row=2, column=3, padx=(6, 0), sticky="nsew")

        # Equity curve over sessions
        eq_card = GothicCard(body)
        eq_card.grid(row=3, column=0, columnspan=4, sticky="nsew",
                      pady=(12, 12))
        eq_card.grid_columnconfigure(0, weight=1)
        eq_card.grid_rowconfigure(1, weight=1)
        SectionHeader(eq_card, "Equity Over Sessions", glyph=G.RUNE_R).grid(
            row=0, column=0, sticky="ew")
        self.eq_chart = EquityCurve(eq_card, height=240)
        self.eq_chart.grid(row=1, column=0, padx=12, pady=(0, 14), sticky="nsew")

        # Promotion gate readiness
        gate_card = GothicCard(body)
        gate_card.grid(row=4, column=0, columnspan=2, sticky="nsew",
                        padx=(0, 6))
        gate_card.grid_columnconfigure(0, weight=1)
        SectionHeader(gate_card, "Live-Money Gate", glyph=G.SHIELD).grid(
            row=0, column=0, sticky="ew")
        self.gate_beacon = StatusBeacon(gate_card, "—", "neutral")
        self.gate_beacon.grid(row=1, column=0, padx=14, pady=(0, 4), sticky="w")
        self.gate_detail = CodexBox(gate_card, height=180)
        self.gate_detail.grid(row=2, column=0, padx=14, pady=(4, 14), sticky="nsew")

        # Model status
        model_card = GothicCard(body)
        model_card.grid(row=4, column=2, columnspan=2, sticky="nsew",
                         padx=(6, 0))
        model_card.grid_columnconfigure(0, weight=1)
        SectionHeader(model_card, "Trained Model", glyph=G.RUNE_F).grid(
            row=0, column=0, sticky="ew")
        self.model_beacon = StatusBeacon(model_card, "Unforged", "neutral")
        self.model_beacon.grid(row=1, column=0, padx=14, pady=(0, 4), sticky="w")
        self.model_detail = CodexBox(model_card, height=180)
        self.model_detail.grid(row=2, column=0, padx=14, pady=(4, 14), sticky="nsew")

        # Postmortem breakdown
        lessons_card = GothicCard(body)
        lessons_card.grid(row=5, column=0, columnspan=4, sticky="nsew",
                           pady=(12, 0))
        lessons_card.grid_columnconfigure(0, weight=1)
        SectionHeader(lessons_card, "Codex Growth", glyph=G.CROSS).grid(
            row=0, column=0, sticky="ew")
        self.lessons_detail = CodexBox(lessons_card, height=200)
        self.lessons_detail.grid(row=1, column=0, padx=14, pady=(0, 14),
                                  sticky="nsew")

        # Profit / risk optimizer panel
        opt_card = GothicCard(body)
        opt_card.grid(row=6, column=0, columnspan=4, sticky="nsew",
                       pady=(12, 0))
        opt_card.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(opt_card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        head.grid_columnconfigure(0, weight=1)
        SectionHeader(head, "Profit / Risk Optimizer  (self-learning)",
                       glyph=G.RUNE_T).grid(row=0, column=0, sticky="w")
        from brzrkr_app.widgets import RuneButton
        RuneButton(head, "Optimise Now", glyph=G.EXEC,
                    command=self._run_optimizer
                    ).grid(row=0, column=1, padx=(0, 14), pady=(10, 4),
                            sticky="e")
        self.opt_detail = CodexBox(opt_card, height=200)
        self.opt_detail.grid(row=1, column=0, padx=14, pady=(0, 14),
                              sticky="nsew")

        # ================================================================
        # TRADING TYPE STRATEGIES — 4 separated sections
        # ================================================================

        # ── Section header ──────────────────────────────────────────────
        InkDivider(body, length=720).grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(20, 4))
        ctk.CTkLabel(
            body,
            text=f"  {G.CROSS}  TRADING TYPE PLAYBOOKS",
            font=ctk.CTkFont(family=FONT_DISPLAY[0], size=15, weight="bold"),
            text_color=C.SIGIL, anchor="w",
        ).grid(row=8, column=0, columnspan=4, sticky="ew", padx=4, pady=(0, 8))

        # ── 1. SWING TRADING ────────────────────────────────────────────
        sw_card = GothicCard(body)
        sw_card.grid(row=9, column=0, columnspan=2, sticky="nsew",
                      padx=(0, 6), pady=(0, 8))
        sw_card.grid_columnconfigure(0, weight=1)
        SectionHeader(sw_card, "Swing Trading", glyph=G.RUNE_O).grid(
            row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            sw_card,
            text=(
                f"  {G.RIGHT}  Hold: 2–10 days  ·  Target: +3–8%  ·  Stop: -1.5–2%\n"
                f"  {G.RIGHT}  Entry: MACD bullish cross above SMA20  +  RSI 40–60\n"
                f"  {G.RIGHT}  Vol: ratio > 1.3× (institution interest confirmation)\n"
                f"  {G.RIGHT}  Instruments: stocks, sector ETFs, index ETFs\n"
                f"  {G.RIGHT}  Size: 5% equity per trade  ·  Max 4 concurrent positions\n"
                f"  {G.RIGHT}  Exit: TP hit  OR  MACD bear cross  OR  close < SMA20"
            ),
            justify="left", anchor="w",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
        ).grid(row=1, column=0, padx=14, pady=(4, 4), sticky="ew")
        self._sw_picks_box = CodexBox(sw_card, height=110)
        self._sw_picks_box.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")

        # ── 2. FUTURES TRADING ──────────────────────────────────────────
        ft_card = GothicCard(body)
        ft_card.grid(row=9, column=2, columnspan=2, sticky="nsew",
                      padx=(6, 0), pady=(0, 8))
        ft_card.grid_columnconfigure(0, weight=1)
        SectionHeader(ft_card, "Futures Trading", glyph=G.RUNE_T).grid(
            row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            ft_card,
            text=(
                f"  {G.RIGHT}  Proxies: GLD (gold) · USO (crude) · TLT (bonds) · UNG (gas)\n"
                f"  {G.RIGHT}  Leveraged: TQQQ/SQQQ · UPRO/SPXS · TNA/TZA\n"
                f"  {G.RIGHT}  Hold: hours to 3 days  ·  Target: +2–5%  ·  Stop: -1–2%\n"
                f"  {G.RIGHT}  Entry: trend alignment + volume surge + regime confirm\n"
                f"  {G.RIGHT}  Regime: trending_up = long leveraged, volatile = avoid\n"
                f"  {G.RIGHT}  Size: 3% equity (leveraged) / 5% equity (proxy ETF)\n"
                f"  {G.RIGHT}  Note: true /ES /NQ require CME feed — use ETF proxies"
            ),
            justify="left", anchor="w",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
        ).grid(row=1, column=0, padx=14, pady=(4, 4), sticky="ew")
        self._ft_picks_box = CodexBox(ft_card, height=110)
        self._ft_picks_box.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")

        # ── 3. OPTIONS TRADING ──────────────────────────────────────────
        op_card = GothicCard(body)
        op_card.grid(row=10, column=0, columnspan=2, sticky="nsew",
                      padx=(0, 6), pady=(0, 8))
        op_card.grid_columnconfigure(0, weight=1)
        SectionHeader(op_card, "Options Trading", glyph=G.EXEC).grid(
            row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            op_card,
            text=(
                f"  {G.RIGHT}  Setups: vol-crush plays · oversold bounces · breakout calls\n"
                f"  {G.RIGHT}  Entry: RSI < 35 (calls) or RSI > 65 (puts) + vol surge\n"
                f"  {G.RIGHT}  Target: +40–100%  ·  Max loss: -50%  ·  Hold: 1–5 days\n"
                f"  {G.RIGHT}  Strike: ATM or 1 strike OTM  ·  Expiry: 7–21 DTE\n"
                f"  {G.RIGHT}  Size: 1–2% equity per trade (options risk is defined)\n"
                f"  {G.RIGHT}  Symbols: SPY, QQQ, TSLA, NVDA, AAPL, META, AMD\n"
                f"  {G.RIGHT}  Avoid: earnings within 3 days (IV crush risk)"
            ),
            justify="left", anchor="w",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
        ).grid(row=1, column=0, padx=14, pady=(4, 4), sticky="ew")
        self._op_picks_box = CodexBox(op_card, height=110)
        self._op_picks_box.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")

        # ── 4. PENNY STOCK TRADING ───────────────────────────────────────
        ps_card = GothicCard(body)
        ps_card.grid(row=10, column=2, columnspan=2, sticky="nsew",
                      padx=(6, 0), pady=(0, 8))
        ps_card.grid_columnconfigure(0, weight=1)
        SectionHeader(ps_card, "Penny Stock Trading", glyph=G.SKULL).grid(
            row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            ps_card,
            text=(
                f"  {G.RIGHT}  Definition: price < $5  ·  Vol surge 2×+ baseline\n"
                f"  {G.RIGHT}  Entry: price breaks 20d high + volume > 2× avg\n"
                f"  {G.RIGHT}  Target: +15–50%  ·  Hard stop: -8%  ·  Hold: hours to 2 days\n"
                f"  {G.RIGHT}  Size: MAX 1% equity (extreme vol = small size)\n"
                f"  {G.RIGHT}  Avoid: stocks with < 500k avg daily volume\n"
                f"  {G.RIGHT}  Watch: SOUN · BBAI · QBTS · RGTI · ARQQ · HIMS · DJT\n"
                f"  {G.RIGHT}  News catalyst required — do NOT trade on technicals alone"
            ),
            justify="left", anchor="w",
            text_color=C.PARCHMENT,
            font=ctk.CTkFont(family=FONT_MONO[0], size=10),
        ).grid(row=1, column=0, padx=14, pady=(4, 4), sticky="ew")
        self._ps_picks_box = CodexBox(ps_card, height=110)
        self._ps_picks_box.grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")

    # ------------------------------------------------------------------
    def update_from(self, snap: dict) -> None:
        self._refresh()

    def _refresh(self) -> None:
        # Track record → equity series
        sessions = self._read_track_record()
        if sessions:
            equity = [s.get("end_equity", 0.0) for s in sessions]
            labels = [s.get("ts", "")[:10] for s in sessions]
            self.eq_chart.set_data(equity, labels)
            wins = sum(1 for s in sessions if s.get("pnl_pct", 0) > 0)
            total = len(sessions)
            wr = wins / total * 100 if total else 0
            first = sessions[0].get("start_equity") or sessions[0].get("end_equity", 100_000)
            cum_pnl = (equity[-1] - first) if equity else 0
            cum_pct = cum_pnl / first * 100 if first else 0

            self.m_sessions.set(str(total))
            self.m_pnl.set(f"${cum_pnl:+,.2f}",
                            sub=f"{cum_pct:+.2f}%",
                            color=pnl_color(cum_pnl))
            self.m_winrate.set(f"{wr:.1f}%",
                                color=C.LIFE if wr >= 50 else C.WOUND)
        else:
            self.eq_chart.set_data([])
            self.m_sessions.set("0", sub="no sessions yet")
            self.m_pnl.set("—")
            self.m_winrate.set("—")

        # Promotion gate readiness
        try:
            from src.execution.promotion_gate import PromotionGate
            verdict = PromotionGate().evaluate()
            if verdict["passed"]:
                self.gate_beacon.set("READY for live money", "ok")
                detail = (
                    f"  {G.DOT_ON}  All criteria met.\n\n"
                    f"  Stats:\n"
                )
            else:
                self.gate_beacon.set("LOCKED — paper only", "warn")
                detail = (
                    f"  {G.SHIELD}  {verdict['reason']}\n\n"
                    f"  Stats:\n"
                )
            stats = verdict.get("stats", {})
            for k, v in stats.items():
                detail += f"    {k:<26s}  {v}\n"
            self.gate_detail.set_text(detail)
        except Exception as exc:  # noqa: BLE001
            self.gate_detail.set_text(f"  {G.SKULL}  gate read failed: {exc}")

        # Model
        if MODEL_FILE.exists():
            kb = MODEL_FILE.stat().st_size / 1024
            mtime = datetime.fromtimestamp(MODEL_FILE.stat().st_mtime,
                                              tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - mtime).days
            state = "ok" if age_days < 30 else "warn"
            self.model_beacon.set(f"Forged ({age_days}d ago)", state)
            text = (
                f"  {G.RUNE_F}  models/xgb.json  ({kb:.1f} KB)\n"
                f"     trained: {mtime.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            )
            if MODEL_REPORT.exists():
                try:
                    rep = json.loads(MODEL_REPORT.read_text())
                    text += (
                        f"  CV accuracy mean:   {rep.get('cv_scores') and sum(rep['cv_scores'])/len(rep['cv_scores']):.4f}\n"
                        f"  Log-loss mean:      {rep.get('cv_logloss') and sum(rep['cv_logloss'])/len(rep['cv_logloss']):.4f}\n"
                        f"  Samples / features: {rep.get('n_samples', 0)} / {rep.get('n_features', 0)}\n"
                        f"  Label distribution: {rep.get('label_distribution', {})}\n\n"
                        f"  Top 5 features by gain:\n"
                    )
                    imp = rep.get("feature_importance", {})
                    for k, v in list(imp.items())[:5]:
                        text += f"     {k:<24s}  {v:>7.2f}\n"
                except Exception as exc:  # noqa: BLE001
                    text += f"\n  (report parse failed: {exc})"
            else:
                text += f"  (no xgb_report.json found)\n"
            self.model_detail.set_text(text)
        else:
            self.model_beacon.set("Unforged — heuristic only", "warn",
                                    glyph=G.DOT_OFF)
            self.model_detail.set_text(
                f"  {G.HOURGLASS}  No trained model found.\n\n"
                f"  Forge one:\n"
                f"     python train.py --symbols SPY QQQ AAPL --lookback-days 1825\n\n"
                f"  After training, the agent auto-loads models/xgb.json on next launch."
            )

        # Postmortem lessons growth
        try:
            from src.learning.postmortem_db import PostmortemDB
            db = PostmortemDB()
            db.bootstrap_if_empty()
            stats = db.stats()
            self.m_lessons.set(str(stats["total"]),
                                sub=f"{stats['confirmed_by_observation']} confirmed")
            lines = [
                f"  {G.CROSS}  Total lessons:        {stats['total']}",
                f"  {G.RUNE_T}  Self-confirmed:      {stats['confirmed_by_observation']}",
                "",
                "  By source:",
            ]
            for src, n in sorted(stats["by_source"].items(), key=lambda kv: -kv[1]):
                lines.append(f"     {src:<14s}  {n}")
            lines.append("")
            lines.append("  By severity:")
            for sev in sorted(stats["by_severity"].keys()):
                bars = G.DOT_ON * sev
                lines.append(f"     {bars:<6s} ({sev})    {stats['by_severity'][sev]}")
            self.lessons_detail.set_text("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            self.lessons_detail.set_text(f"  {G.SKULL}  codex read failed: {exc}")

        # Optimizer status
        self._refresh_optimizer()

        # Strategy type picks
        self._refresh_strategy_picks()

    def _read_track_record(self) -> list[dict]:
        if not TRACK_FILE.exists():
            return []
        out = []
        for ln in TRACK_FILE.read_text().splitlines():
            ln = ln.strip()
            if not ln: continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    # ------------------------------------------------------------------
    def _refresh_optimizer(self) -> None:
        try:
            from src.learning.profit_optimizer import load_learned_params
            p = load_learned_params()
            if p is None:
                self.opt_detail.set_text(
                    f"  {G.HOURGLASS}  No learned parameters yet.\n\n"
                    f"  Click 'Optimise Now' to scan past trades and\n"
                    f"  compute the optimal stop / take-profit ratios.\n\n"
                    f"  Or run from CLI:\n"
                    f"     python learn.py optimize           # dry-run\n"
                    f"     python learn.py optimize --apply   # commit"
                )
                return
            lines = [
                f"  {G.RUNE_T}  Last optimised:    {p.get('computed_at', '?')[:19]}",
                f"  {G.RIGHT}  Trades analysed:    {p.get('n_trades', 0)}",
                f"  {G.RIGHT}  Win rate:           {p.get('win_rate', 0) * 100:.2f}%",
                f"  {G.RIGHT}  Expected R/trade:   {p.get('expected_R', 0):+.4f}",
                f"  {G.RIGHT}  Profit factor:      {p.get('profit_factor', 0):.2f}",
                f"  {G.RIGHT}  Sharpe-like:        {p.get('sharpe_proxy', 0):.2f}",
                "",
                f"  {G.RUNE_F}  Recommended parameters",
                f"     stop %:         {p.get('optimal_stop_pct', 0) * 100:.2f}%",
                f"     take-profit %:  {p.get('optimal_tp_pct', 0) * 100:.2f}%",
                f"     R:R ratio:      {p.get('optimal_rr_ratio', 2.0)}",
                "",
                f"  {G.RIGHT}  notes: {p.get('notes', '')}",
            ]
            by_dir = p.get("by_direction") or {}
            if by_dir:
                lines.append("")
                lines.append(f"  {G.RUNE_R}  By direction")
                for d, s in by_dir.items():
                    lines.append(
                        f"     {d:<5s}  n={s.get('n_trades', 0):>4d}  "
                        f"win={s.get('win_rate', 0)*100:.1f}%  "
                        f"EV={s.get('expected_R', 0):+.3f} R"
                    )
            self.opt_detail.set_text("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            self.opt_detail.set_text(f"  {G.SKULL}  read failed: {exc}")

    def _run_optimizer(self) -> None:
        try:
            from src.learning.profit_optimizer import ProfitOptimizer
            opt = ProfitOptimizer()
            result = opt.analyze()
            opt.write(result)
            self.app.toast(f"{G.RUNE_T}  Optimised over {result.n_trades} trades.")
            self._refresh_optimizer()
        except Exception as exc:  # noqa: BLE001
            self.app.toast(f"Optimise failed: {exc}")

    def _refresh_strategy_picks(self) -> None:
        """Load live top picks per strategy type from VectorStore cache."""
        try:
            from src.data.vector_store import VectorStore
            vs    = VectorStore()
            snaps = vs.load_latest()
            if not snaps:
                msg = "  No market data yet — collector running in background."
                for box in (self._sw_picks_box, self._ft_picks_box,
                            self._op_picks_box, self._ps_picks_box):
                    box.set_text(msg)
                return

            def _fmt(s: dict, extra: str = "") -> str:
                sig   = "↑" if s.get("signal") == "bullish" else "↓"
                score = s.get("score", 0)
                price = s.get("price", 0)
                chg   = s.get("change_1d_pct", 0)
                arrow = "▲" if chg >= 0 else "▼"
                note  = s.get("note", "")[:45]
                return (f"  {sig} {s.get('symbol',''):<6}  "
                        f"${price:>8,.2f}  {arrow}{abs(chg):.1f}%  "
                        f"score={score:>+4.0f}  {note}")

            # Swing: stocks/ETFs with MACD cross and momentum
            _SW = {"STOCK", "INDEX_ETF", "SECTOR_ETF"}
            sw_picks = sorted(
                [s for s in snaps
                 if s.get("asset_type") in _SW
                 and (s.get("macd_signal") != "flat" or abs(s.get("momentum_5d", 0)) > 2)],
                key=lambda s: abs(s.get("score", 0)), reverse=True)[:6]
            self._sw_picks_box.set_text(
                "\n".join(_fmt(s) for s in sw_picks) if sw_picks
                else "  No swing setups currently scored.")

            # Futures: commodity/leveraged/bond ETFs
            _FT = {"COMMODITY_ETF", "BOND_ETF", "LEVERAGED_ETF", "VOLATILITY"}
            ft_picks = sorted(
                [s for s in snaps if s.get("asset_type") in _FT],
                key=lambda s: abs(s.get("score", 0)), reverse=True)[:6]
            self._ft_picks_box.set_text(
                "\n".join(_fmt(s) for s in ft_picks) if ft_picks
                else "  No futures proxy setups currently scored.")

            # Options: extreme RSI or vol surge, liquid names
            def _op_score(s):
                return abs(s.get("rsi", 50) - 50) + s.get("vol_ratio", 1) * 5
            op_picks = sorted(
                [s for s in snaps
                 if s.get("rsi", 50) < 38 or s.get("rsi", 50) > 62
                 or s.get("vol_ratio", 1) > 1.5],
                key=_op_score, reverse=True)[:6]
            self._op_picks_box.set_text(
                "\n".join(_fmt(s) for s in op_picks) if op_picks
                else "  No options setups currently scored.")

            # Penny: price < $5 or SMALL_CAP
            ps_picks = sorted(
                [s for s in snaps
                 if s.get("price", 999) < 5.0 or s.get("asset_type") == "SMALL_CAP"],
                key=lambda s: s.get("vol_ratio", 1) * abs(s.get("score", 0)),
                reverse=True)[:6]
            self._ps_picks_box.set_text(
                "\n".join(_fmt(s) for s in ps_picks) if ps_picks
                else "  No penny/small-cap picks currently scored.")

        except Exception as exc:
            msg = f"  Picks unavailable: {exc}"
            for box in (self._sw_picks_box, self._ft_picks_box,
                        self._op_picks_box, self._ps_picks_box):
                try:
                    box.set_text(msg)
                except Exception:
                    pass
