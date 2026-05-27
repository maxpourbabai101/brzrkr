"""CLI for the self-updating postmortem knowledge base.

Subcommands:
    init      Bootstrap the DB with seed knowledge (idempotent).
    stats     Show counts by category / severity / source.
    list      List lessons, optionally filtered by category / severity / source.
    show      Print one lesson in full detail.
    add       Manually add a new lesson (prompts for fields).
    preflight Run preflight checks for `agent` or `train`.
    export    Dump the DB as Markdown to docs/postmortems.md.

The DB lives at `data/postmortems.jsonl`. Each line is one Lesson
JSON record. The DB is auto-bootstrapped from seed_knowledge.py on
first use.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.correlation_analyzer import CorrelationAnalyzer
from src.learning.postmortem_db import Lesson, PostmortemDB
from src.learning.preflight import run_preflight
from src.learning.profit_optimizer import ProfitOptimizer, load_learned_params

logger = logging.getLogger("trading_enhancer.learn")


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run the profit-optimiser over historical trades."""
    opt = ProfitOptimizer()
    result = opt.analyze()
    print()
    print(f"=== Profit / Risk Optimisation ===")
    print(f"Trades analysed     : {result.n_trades}")
    print(f"Win rate            : {result.win_rate * 100:.2f}%")
    print(f"Expected R          : {result.expected_R:+.4f}")
    print(f"Profit factor       : {result.profit_factor:.2f}")
    print(f"Avg win  R          : {result.avg_win_R:+.4f}")
    print(f"Avg loss R          : {result.avg_loss_R:+.4f}")
    print(f"Sharpe-like         : {result.sharpe_proxy:.2f}")
    print()
    print(f"Optimal stop %      : {result.optimal_stop_pct * 100:.2f}%")
    print(f"Optimal take-profit : {result.optimal_tp_pct * 100:.2f}%")
    print(f"Optimal R:R ratio   : {result.optimal_rr_ratio}")
    print()
    if result.by_direction:
        print(f"By direction:")
        for d, s in result.by_direction.items():
            print(f"  {d:5s}  n={s['n_trades']:>4d}  "
                  f"win={s['win_rate']*100:.1f}%  EV={s['expected_R']:+.3f} R")
    print()
    print(f"Notes: {result.notes}")
    if args.apply:
        path = opt.write(result)
        print(f"\nWrote: {path}")
    else:
        print(f"\n(dry-run; pass --apply to write data/learned_params.json)")
    return 0


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    db = PostmortemDB()
    n = db.bootstrap_if_empty()
    if n == 0:
        print(f"DB already exists at {db.path} with {len(db.all_lessons)} lessons.")
    else:
        print(f"Seeded {n} lessons → {db.path}.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    db = PostmortemDB()
    db.bootstrap_if_empty()
    s = db.stats()
    print(f"Postmortem DB ({db.path})")
    print(f"  Total lessons     : {s['total']}")
    print(f"  Self-confirmed    : {s['confirmed_by_observation']} "
          f"(lessons that the observer has matched against this system's behavior)")
    print()
    print(f"  By category:")
    for cat, n in sorted(s["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"    {cat:18s} {n}")
    print()
    print(f"  By severity:")
    for sev in sorted(s["by_severity"].keys()):
        n = s["by_severity"][sev]
        print(f"    {'🔴' * sev:6s} (sev {sev})  {n}")
    print()
    print(f"  By source:")
    for src, n in sorted(s["by_source"].items(), key=lambda kv: -kv[1]):
        print(f"    {src:12s} {n}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    db = PostmortemDB()
    db.bootstrap_if_empty()
    lessons = db.find(
        category=args.category,
        tag=args.tag,
        min_severity=args.min_severity,
        source=args.source,
    )
    if not lessons:
        print("No lessons match.")
        return 0
    for l in lessons[: args.limit]:
        sev = "🔴" * l.severity + "⚪" * (5 - l.severity)
        cnt = f" ×{l.confirmed_count}" if l.confirmed_count else ""
        print(f"  {sev}  [{l.id:8s}]{cnt:5s} {l.category:14s} {l.title}")
    print(f"\n{len(lessons)} match · showing {min(len(lessons), args.limit)}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    db = PostmortemDB()
    db.bootstrap_if_empty()
    l = db.get(args.id)
    if l is None:
        print(f"No lesson with id {args.id!r}.", file=sys.stderr)
        return 1
    print()
    print(f"=== {l.title} ===")
    print(f"id          : {l.id}")
    print(f"category    : {l.category}")
    print(f"severity    : {'🔴' * l.severity} ({l.severity}/5)")
    print(f"source      : {l.source}")
    print(f"confirmed   : {l.confirmed_count}× by observer")
    print(f"first seen  : {l.first_seen}")
    print(f"last confirm: {l.last_confirmed or '—'}")
    if l.tags:
        print(f"tags        : {', '.join(l.tags)}")
    print()
    print("Description :")
    print("   ", l.description)
    print()
    print("Symptom     :")
    print("   ", l.symptom)
    print()
    print("Mitigation  :")
    print("   ", l.mitigation)
    if l.references:
        print()
        print("References  :")
        for ref in l.references:
            print(f"   - {ref}")
    print()
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Manual entry. Each field can be passed via flag or interactively."""
    def ask(prompt: str, default: str = "") -> str:
        text = input(f"  {prompt}{f' [{default}]' if default else ''}: ").strip()
        return text or default

    print("Add a new lesson (Ctrl-C to abort)")
    lid = args.id or ask("id (e.g. user_my_lesson)")
    if not lid:
        print("Aborted.", file=sys.stderr)
        return 1
    lesson = Lesson(
        id=lid,
        category=args.category or ask("category"),
        title=args.title or ask("title"),
        description=args.description or ask("description"),
        symptom=args.symptom or ask("symptom"),
        mitigation=args.mitigation or ask("mitigation"),
        severity=int(args.severity or ask("severity (1-5)", "3")),
        source="user",
        first_seen=datetime.now(timezone.utc).isoformat(),
        tags=[t.strip() for t in (args.tags or ask("tags (comma-separated)", "")).split(",") if t.strip()],
        references=[r.strip() for r in (args.references or "").split(",") if r.strip()],
    )
    db = PostmortemDB()
    is_new = db.add_or_update(lesson)
    print(f"{'Added' if is_new else 'Updated'} {lesson.id}.")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    report = run_preflight(args.context, live_money=args.live_money)
    print(report.to_text())
    return 0 if report.passed else 2


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run conditional-P&L analysis on the track record and report or
    apply severity adjustments."""
    analyzer = CorrelationAnalyzer()
    report = analyzer.analyze(
        min_per_group=args.min_per_group,
        alpha=args.alpha,
        meaningful_effect=args.meaningful_effect,
    )
    print(report.to_text())

    changes = analyzer.apply_recommendations(report, apply=args.apply)
    if changes:
        print("\n--- Severity changes ---")
        for ch in changes:
            print(f"  {'(applied)' if args.apply else '(dry-run)'}  {ch}")
        if not args.apply:
            print("\nRun again with --apply to commit these severity changes.")
    else:
        print("\nNo severity changes suggested.")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    db = PostmortemDB()
    db.bootstrap_if_empty()
    md = db.to_markdown(top=args.top)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Exported {len(db.all_lessons)} lessons → {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Self-updating postmortem DB CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="Seed the DB with bundled knowledge")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("stats", help="Counts by category / severity / source")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("list", help="List lessons (filterable)")
    p.add_argument("--category")
    p.add_argument("--tag")
    p.add_argument("--source")
    p.add_argument("--min-severity", type=int, default=1)
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="Print one lesson in full")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("add", help="Add a lesson manually (interactive)")
    for f in ("id", "category", "title", "description", "symptom",
              "mitigation", "severity", "tags", "references"):
        p.add_argument(f"--{f}")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("preflight", help="Run preflight checks")
    p.add_argument("context", choices=["agent", "train"])
    p.add_argument("--live-money", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("analyze",
                        help="Conditional-P&L analysis: which lessons actually correlate with outcomes?")
    p.add_argument("--min-per-group", type=int, default=5,
                   help="Minimum sessions per fired/unfired group (default 5).")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="p-value cutoff for 'significant' (default 0.10).")
    p.add_argument("--meaningful-effect", type=float, default=0.001,
                   help="Minimum |Δ pnl_pct| to treat as non-neutral (default 0.001 = 0.1%%).")
    p.add_argument("--apply", action="store_true",
                   help="Commit suggested severity changes. Default = dry-run.")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("export", help="Dump DB as Markdown")
    p.add_argument("--output", default="docs/postmortems.md")
    p.add_argument("--top", type=int, default=100)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("optimize",
                        help="Tune stop / take-profit ratios from past trades")
    p.add_argument("--apply", action="store_true",
                   help="Write data/learned_params.json. Default = dry-run.")
    p.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
