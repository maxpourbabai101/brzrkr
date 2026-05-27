"""Curated library of historical market scenarios for backtesting.

Each scenario is a real, dated episode. Categories cover the full
spectrum of conditions a strategy needs to survive:

* ``crash``         — sharp drawdown, vol spike
* ``rally``         — sustained uptrend
* ``vol_spike``     — IV explosion without long-lasting trend
* ``regime_change`` — structural shift (rate cycle, policy)
* ``grinding``      — boring sideways / steady trend
* ``crisis``        — multi-asset stress
* ``squeeze``       — short-squeeze / meme dynamics
* ``post_event``    — immediate-aftermath of FOMC, earnings, etc.

Use :func:`all_scenarios` to get the full battery, or :func:`by_category`
to filter. The :func:`benchmark_summary` helper renders a quick view
of buy-and-hold context per scenario for sanity comparison.

Sources: Wikipedia event timelines + market data cross-references.
Date ranges are inclusive (yfinance period1/period2 semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass(frozen=True)
class MarketScenario:
    name: str                       # unique identifier
    category: str
    symbols: List[str]              # which tickers to replay through
    start: datetime
    end: datetime
    description: str
    # Optional context for the report — what buy-and-hold would have done.
    benchmark_symbol: str = "SPY"
    expected_difficulty: str = "moderate"   # "easy" / "moderate" / "hard" / "brutal"

    @property
    def days(self) -> int:
        return (self.end - self.start).days


def _d(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------
SCENARIOS: List[MarketScenario] = [
    # ============ CRASHES ============
    MarketScenario(
        name="covid_crash_2020",
        category="crash",
        symbols=["SPY", "QQQ", "IWM"],
        start=_d(2020, 2, 18), end=_d(2020, 4, 8),
        description="COVID-19 crash & initial recovery. SPY -34% then +20% in 50 days. Liquidity vacuum, circuit breakers triggered.",
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="sept_2008_lehman",
        category="crisis",
        symbols=["SPY", "XLF"],
        start=_d(2008, 9, 1), end=_d(2008, 10, 31),
        description="Lehman bankruptcy + TARP. Financial crisis epicenter. SPY -27% in two months.",
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="dec_2018_selloff",
        category="crash",
        symbols=["SPY", "QQQ"],
        start=_d(2018, 10, 1), end=_d(2018, 12, 31),
        description="Q4 2018 Fed hike + trade war sell-off. SPY -19% peak-to-trough into Christmas Eve.",
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="aug_2015_china_crash",
        category="crash",
        symbols=["SPY", "FXI"],
        start=_d(2015, 8, 1), end=_d(2015, 9, 30),
        description="China devaluation, SPY -11% in 5 days. Multiple 1000-point Dow swings.",
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="svb_collapse_2023",
        category="crisis",
        symbols=["SPY", "XLF", "KRE"],
        start=_d(2023, 3, 6), end=_d(2023, 3, 24),
        description="SVB / Signature / Credit Suisse bank failures. Regional bank index KRE -28%.",
        expected_difficulty="brutal",
    ),

    # ============ VOL SPIKES ============
    MarketScenario(
        name="volmageddon_feb_2018",
        category="vol_spike",
        symbols=["SPY", "VXX"],
        start=_d(2018, 1, 25), end=_d(2018, 2, 16),
        description="VIX 17→50 in two days. XIV ETN collapsed (-96%), terminated. Short-vol trade extinct overnight.",
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="covid_vol_march_2020",
        category="vol_spike",
        symbols=["SPY", "VXX"],
        start=_d(2020, 3, 9), end=_d(2020, 3, 23),
        description="VIX hit 82 (record close). 4 circuit-breaker halts in 10 sessions.",
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="brexit_vote_2016",
        category="vol_spike",
        symbols=["SPY", "EWU"],
        start=_d(2016, 6, 20), end=_d(2016, 7, 1),
        description="Brexit referendum. GBP -8% overnight. SPY -5% then full recovery in 8 days.",
        expected_difficulty="hard",
    ),

    # ============ RALLIES ============
    MarketScenario(
        name="covid_rebound_2020",
        category="rally",
        symbols=["SPY", "QQQ", "ARKK"],
        start=_d(2020, 4, 1), end=_d(2020, 8, 31),
        description="Post-COVID liquidity rally. QQQ +50%, ARKK +90%. Hard for shorts.",
        expected_difficulty="moderate",
    ),
    MarketScenario(
        name="ai_rally_2023",
        category="rally",
        symbols=["NVDA", "MSFT", "META", "QQQ"],
        start=_d(2023, 1, 1), end=_d(2023, 7, 31),
        description="AI-led mega-cap rally. NVDA +200%, META +160%. Concentrated leadership.",
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="trump_tax_rally_late_2017",
        category="rally",
        symbols=["SPY", "QQQ"],
        start=_d(2017, 10, 1), end=_d(2017, 12, 31),
        description="Tax-cut rally. SPY +6% Q4, near-zero vol days, low pullback depth.",
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="q1_2023_rally",
        category="rally",
        symbols=["SPY", "QQQ"],
        start=_d(2023, 1, 3), end=_d(2023, 3, 31),
        description="January 2023 rally, then bank crisis stall. SPY +7%.",
        expected_difficulty="moderate",
    ),

    # ============ REGIME CHANGES ============
    MarketScenario(
        name="2022_rate_hike_shock",
        category="regime_change",
        symbols=["SPY", "TLT", "QQQ"],
        start=_d(2022, 1, 1), end=_d(2022, 6, 30),
        description="Fed pivot from zero rates. TLT -22% in 6mo, QQQ -29%. Bond/stock correlation broken.",
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="zero_rate_2020",
        category="regime_change",
        symbols=["SPY", "TLT"],
        start=_d(2020, 4, 1), end=_d(2020, 12, 31),
        description="Zero-rate regime onset. QE Forever. Low realized vol after the March spike.",
        expected_difficulty="easy",
    ),

    # ============ SQUEEZES ============
    MarketScenario(
        name="gamestop_jan_2021",
        category="squeeze",
        symbols=["GME", "AMC", "SPY"],
        start=_d(2021, 1, 11), end=_d(2021, 2, 12),
        description="GME +1700% in 2 weeks. AMC, BBBY, KOSS dragged along. Melvin Capital lost 53% in January.",
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="ai_meme_summer_2023",
        category="squeeze",
        symbols=["AI", "SOUN", "BBAI"],
        start=_d(2023, 5, 1), end=_d(2023, 7, 31),
        description="AI-themed micro-caps spiked on hype. AI 3x off lows. Subsequently gave most back.",
        expected_difficulty="hard",
    ),

    # ============ GRINDING / BORING (essential — most days are like this) ============
    MarketScenario(
        name="grind_h1_2017",
        category="grinding",
        symbols=["SPY", "QQQ"],
        start=_d(2017, 1, 1), end=_d(2017, 6, 30),
        description="Low-vol grind upward. VIX averaged 11. Few days with any directional move.",
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="grind_h2_2019",
        category="grinding",
        symbols=["SPY", "QQQ", "IWM"],
        start=_d(2019, 7, 1), end=_d(2019, 12, 31),
        description="Steady second-half uptrend. SPY +9%. Light news, easy regime.",
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="grind_h1_2024",
        category="grinding",
        symbols=["SPY", "QQQ"],
        start=_d(2024, 1, 1), end=_d(2024, 6, 30),
        description="AI-led grind to new highs. SPY +15% H1, modest pullbacks only.",
        expected_difficulty="easy",
    ),

    # ============ POST-EVENT ============
    MarketScenario(
        name="post_election_nov_2016",
        category="post_event",
        symbols=["SPY", "TLT"],
        start=_d(2016, 11, 7), end=_d(2016, 12, 30),
        description="Trump election reaction: SPY +5%, TLT -8% in weeks. Reflation trade.",
        expected_difficulty="moderate",
    ),
    MarketScenario(
        name="post_fomc_oct_2018",
        category="post_event",
        symbols=["SPY", "TLT"],
        start=_d(2018, 10, 3), end=_d(2018, 10, 12),
        description="Powell 'long way from neutral' speech. SPY -7% in 6 days.",
        expected_difficulty="hard",
    ),
]


def all_scenarios() -> List[MarketScenario]:
    return list(SCENARIOS)


def by_category(category: str) -> List[MarketScenario]:
    return [s for s in SCENARIOS if s.category == category]


def by_name(name: str) -> Optional[MarketScenario]:
    for s in SCENARIOS:
        if s.name == name:
            return s
    return None


def categories() -> List[str]:
    return sorted({s.category for s in SCENARIOS})


def benchmark_summary() -> str:
    """Quick text summary of the library."""
    by_cat: dict = {}
    for s in SCENARIOS:
        by_cat.setdefault(s.category, []).append(s)
    lines = [f"Scenario library: {len(SCENARIOS)} episodes across {len(by_cat)} categories", ""]
    for cat in sorted(by_cat):
        lines.append(f"  {cat.upper()}  ({len(by_cat[cat])})")
        for s in by_cat[cat]:
            lines.append(f"    • {s.name:<30}  {s.start.date()} → {s.end.date()}  "
                         f"({s.days:>3}d)  [{s.expected_difficulty}]")
    return "\n".join(lines)
