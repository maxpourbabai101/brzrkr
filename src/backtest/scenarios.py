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
* ``sector``        — sector-specific rotation / cycle

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

    # ============================================================
    # CRASHES
    # ============================================================

    MarketScenario(
        name="dot_com_crash_2000",
        category="crash",
        symbols=["QQQ", "SPY", "MSFT", "CSCO", "IWM"],
        start=_d(2000, 3, 24), end=_d(2002, 10, 9),
        description=(
            "Dot-com bubble collapse. NASDAQ -78% peak to trough over 2.5 years. "
            "The longest tech bear market on record. QQQ fell from $120 to $20."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="covid_crash_2020",
        category="crash",
        symbols=["SPY", "QQQ", "IWM", "XLF", "GLD"],
        start=_d(2020, 2, 18), end=_d(2020, 4, 8),
        description=(
            "COVID-19 crash and initial recovery. SPY -34% then +20% in 50 days. "
            "Liquidity vacuum, circuit breakers triggered. Fastest 30% bear market ever."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="dec_2018_selloff",
        category="crash",
        symbols=["SPY", "QQQ", "IWM", "TLT"],
        start=_d(2018, 10, 1), end=_d(2018, 12, 31),
        description=(
            "Q4 2018 Fed hike + trade war sell-off. SPY -19% peak-to-trough into "
            "Christmas Eve — worst December since 1931."
        ),
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="aug_2015_china_crash",
        category="crash",
        symbols=["SPY", "FXI", "QQQ", "EEM"],
        start=_d(2015, 8, 1), end=_d(2015, 9, 30),
        description=(
            "China devaluation shock. SPY -11% in 5 days. Multiple 1000-pt Dow swings. "
            "Circuit breakers triggered in Shanghai."
        ),
        expected_difficulty="hard",
    ),

    # ============================================================
    # CRISES
    # ============================================================

    MarketScenario(
        name="great_recession_2007",
        category="crisis",
        symbols=["SPY", "XLF", "QQQ", "TLT", "GLD", "IWM"],
        start=_d(2007, 10, 9), end=_d(2009, 3, 9),
        description=(
            "Global financial crisis full arc — peak to trough. S&P 500 -57%. "
            "Bear Stearns, Fannie/Freddie, Lehman, TARP. The defining stress test."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="sept_2008_lehman",
        category="crisis",
        symbols=["SPY", "XLF", "TLT", "GLD"],
        start=_d(2008, 9, 1), end=_d(2008, 10, 31),
        description=(
            "Lehman bankruptcy + TARP. Financial crisis epicenter. "
            "SPY -27% in two months. XLF -40%. GLD bid as safe haven."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="euro_debt_crisis_2012",
        category="crisis",
        symbols=["SPY", "EFA", "TLT", "GLD", "EWI"],
        start=_d(2012, 1, 1), end=_d(2012, 7, 31),
        description=(
            "European sovereign debt crisis — Spain/Italy yields spike, Draghi 'whatever it takes'. "
            "EWI (Italy ETF) down 30%. Safe-haven flows to GLD and TLT."
        ),
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="svb_collapse_2023",
        category="crisis",
        symbols=["SPY", "XLF", "KRE", "TLT", "GLD"],
        start=_d(2023, 3, 6), end=_d(2023, 3, 24),
        description=(
            "SVB / Signature / Credit Suisse bank failures. Regional bank index "
            "KRE -28%. Treasury rally + bank contagion fear."
        ),
        expected_difficulty="brutal",
    ),

    # ============================================================
    # VOL SPIKES
    # ============================================================

    MarketScenario(
        name="flash_crash_2010",
        category="vol_spike",
        symbols=["SPY", "QQQ", "IWM", "GLD"],
        start=_d(2010, 5, 1), end=_d(2010, 6, 30),
        description=(
            "May 6 flash crash: Dow -1000 pts in minutes (then recovered in 20 min). "
            "First algorithmic liquidity crisis. VIX spiked to 48."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="euro_debt_vol_2011",
        category="vol_spike",
        symbols=["SPY", "EFA", "TLT", "GLD", "VXX"],
        start=_d(2011, 7, 1), end=_d(2011, 9, 30),
        description=(
            "US debt ceiling standoff + European debt panic. S&P downgrade of USA. "
            "SPY -17% in 6 weeks. VIX to 48. Gold to all-time high $1900."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="volmageddon_feb_2018",
        category="vol_spike",
        symbols=["SPY", "VXX", "QQQ", "TLT"],
        start=_d(2018, 1, 25), end=_d(2018, 2, 16),
        description=(
            "VIX 17→50 in two days. XIV ETN collapsed (-96%), terminated. "
            "Short-vol trade extinct overnight. Most violent single-day vol event ever."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="covid_vol_march_2020",
        category="vol_spike",
        symbols=["SPY", "VXX", "TLT", "GLD"],
        start=_d(2020, 3, 9), end=_d(2020, 3, 23),
        description=(
            "VIX hit 82 (record close). 4 circuit-breaker halts in 10 sessions. "
            "Treasuries and gold both sold simultaneously — true liquidity crisis."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="taper_tantrum_2013",
        category="vol_spike",
        symbols=["SPY", "TLT", "EEM", "IEF"],
        start=_d(2013, 5, 1), end=_d(2013, 8, 31),
        description=(
            "Bernanke hints at QE taper. TLT -12% in 3 months. EEM -15%. "
            "First test of market dependence on Fed liquidity."
        ),
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="brexit_vote_2016",
        category="vol_spike",
        symbols=["SPY", "EWU", "EFA", "GLD"],
        start=_d(2016, 6, 20), end=_d(2016, 7, 15),
        description=(
            "Brexit referendum. GBP -8% overnight. SPY -5% then full recovery in 8 days. "
            "Classic one-day vol spike with rapid mean-reversion."
        ),
        expected_difficulty="hard",
    ),

    # ============================================================
    # RALLIES
    # ============================================================

    MarketScenario(
        name="post_crisis_rally_2009",
        category="rally",
        symbols=["SPY", "QQQ", "XLF", "IWM", "AAPL"],
        start=_d(2009, 3, 9), end=_d(2010, 4, 30),
        description=(
            "Post-GFC recovery rally. SPY +80% over 13 months. "
            "XLF doubled. AAPL breakout. Greatest sustained recovery rally in modern history."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="covid_rebound_2020",
        category="rally",
        symbols=["SPY", "QQQ", "ARKK", "TSLA", "NVDA"],
        start=_d(2020, 4, 1), end=_d(2020, 8, 31),
        description=(
            "Post-COVID liquidity rally. QQQ +50%, ARKK +90%, TSLA +300%. "
            "Fastest recovery from a bear market ever. Hard for shorts."
        ),
        expected_difficulty="moderate",
    ),
    MarketScenario(
        name="full_2021_rally",
        category="rally",
        symbols=["SPY", "QQQ", "TSLA", "NVDA", "AAPL", "XLE"],
        start=_d(2021, 1, 1), end=_d(2021, 11, 30),
        description=(
            "Post-vaccine reopening + stimulus rally. SPY +27%. TSLA $700→$1200. "
            "NVDA +130%. XLE +55% as energy reflated. Broad-based bull market."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="ai_rally_2023",
        category="rally",
        symbols=["NVDA", "MSFT", "META", "QQQ", "AAPL"],
        start=_d(2023, 1, 1), end=_d(2023, 7, 31),
        description=(
            "ChatGPT-led AI rally. NVDA +200%, META +160%, MSFT +40%. "
            "Concentrated mega-cap leadership; everything else lagged."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="full_2024",
        category="rally",
        symbols=["SPY", "QQQ", "NVDA", "AAPL", "META", "IWM"],
        start=_d(2024, 1, 1), end=_d(2024, 12, 31),
        description=(
            "2024 AI supercycle + soft-landing narrative. SPY +24%, NVDA +171%. "
            "Rate-cut tailwind. Election volatility but ultimately higher."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="trump_tax_rally_late_2017",
        category="rally",
        symbols=["SPY", "QQQ", "XLF", "IWM"],
        start=_d(2017, 10, 1), end=_d(2017, 12, 31),
        description=(
            "Tax-cut rally. SPY +6% Q4. Near-zero vol days, low pullback depth. "
            "XLF +15% on corporate tax benefit. The perfect grind."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="q1_2023_rally",
        category="rally",
        symbols=["SPY", "QQQ", "XLK", "AAPL"],
        start=_d(2023, 1, 3), end=_d(2023, 3, 31),
        description=(
            "January 2023 bear-market-rally, then SVB stall. SPY +7%. "
            "Tech outperformed as rate fears eased temporarily."
        ),
        expected_difficulty="moderate",
    ),

    # ============================================================
    # REGIME CHANGES
    # ============================================================

    MarketScenario(
        name="zero_rate_onset_2020",
        category="regime_change",
        symbols=["SPY", "TLT", "IEF", "GLD"],
        start=_d(2020, 3, 15), end=_d(2020, 12, 31),
        description=(
            "Fed cuts to zero + QE Infinity. TLT +30% on rate crash, then stable. "
            "New regime: low vol, all assets correlated upward, bonds as tailwind."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="2022_rate_hike_shock",
        category="regime_change",
        symbols=["SPY", "TLT", "QQQ", "XLK", "ARKK"],
        start=_d(2022, 1, 1), end=_d(2022, 6, 30),
        description=(
            "Fed pivot from zero rates. H1 2022: TLT -22%, QQQ -29%, ARKK -55%. "
            "Bond/stock correlation broken. Duration risk repriced globally."
        ),
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="full_bear_2022",
        category="regime_change",
        symbols=["SPY", "QQQ", "TLT", "ARKK", "XLE", "GLD"],
        start=_d(2022, 1, 1), end=_d(2022, 12, 31),
        description=(
            "Full-year 2022 bear market. SPY -19%, QQQ -33%, TLT -29%, ARKK -75%. "
            "XLE the only winner (+64%). Rising rates ended the TINA era."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="oil_crash_regime_2014",
        category="regime_change",
        symbols=["XLE", "XOM", "CVX", "SPY", "IEF"],
        start=_d(2014, 6, 1), end=_d(2016, 1, 31),
        description=(
            "Saudi supply shock collapses crude from $107 to $26. XLE -40%. "
            "XOM -25%. Credit stress in energy HY. 18-month sector bear market."
        ),
        expected_difficulty="hard",
    ),

    # ============================================================
    # GRINDING
    # ============================================================

    MarketScenario(
        name="post_crisis_grind_2010_2011",
        category="grinding",
        symbols=["SPY", "QQQ", "IWM", "XLF"],
        start=_d(2010, 1, 1), end=_d(2011, 4, 30),
        description=(
            "Slow recovery grind after GFC. SPY +15% over 16 months. "
            "Regular 3-5% pullbacks, no sustained trend. Hard to time entries."
        ),
        expected_difficulty="moderate",
    ),
    MarketScenario(
        name="decade_grind_2012_2014",
        category="grinding",
        symbols=["SPY", "QQQ", "IWM", "TLT"],
        start=_d(2012, 1, 1), end=_d(2014, 12, 31),
        description=(
            "Three-year low-vol grind. SPY +60% over 3 years. VIX averaged 14. "
            "The ideal strategy environment — reward patience."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="grind_h1_2017",
        category="grinding",
        symbols=["SPY", "QQQ", "IWM", "XLF"],
        start=_d(2017, 1, 1), end=_d(2017, 6, 30),
        description=(
            "Low-vol grind upward. VIX averaged 11. Few days with any directional move. "
            "Trend-following systems underperformed mean-reversion."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="grind_2019_full",
        category="grinding",
        symbols=["SPY", "QQQ", "IWM", "XLF"],
        start=_d(2019, 1, 1), end=_d(2019, 12, 31),
        description=(
            "Full 2019 trade-war-noise + Fed pivot recovery. SPY +29%. "
            "Many false breakdowns on trade headlines; underlying trend steady up."
        ),
        expected_difficulty="moderate",
    ),
    MarketScenario(
        name="grind_h2_2019",
        category="grinding",
        symbols=["SPY", "QQQ", "IWM"],
        start=_d(2019, 7, 1), end=_d(2019, 12, 31),
        description=(
            "Steady second-half uptrend. SPY +9%. Light news, easy regime. "
            "Classic FOMO grind toward year-end."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="grind_h1_2024",
        category="grinding",
        symbols=["SPY", "QQQ", "NVDA", "AAPL"],
        start=_d(2024, 1, 1), end=_d(2024, 6, 30),
        description=(
            "AI-led grind to new highs. SPY +15% H1, modest pullbacks only. "
            "Concentration risk building in Mag 7."
        ),
        expected_difficulty="easy",
    ),

    # ============================================================
    # SQUEEZES
    # ============================================================

    MarketScenario(
        name="gamestop_jan_2021",
        category="squeeze",
        symbols=["GME", "AMC", "SPY", "QQQ"],
        start=_d(2021, 1, 11), end=_d(2021, 2, 12),
        description=(
            "GME +1700% in 2 weeks. AMC, BBBY, KOSS dragged along. "
            "Melvin Capital -53% in January. WallStreetBets vs. hedge funds."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="ai_meme_summer_2023",
        category="squeeze",
        symbols=["AI", "SOUN", "BBAI", "NVDA"],
        start=_d(2023, 5, 1), end=_d(2023, 7, 31),
        description=(
            "AI-themed micro-caps spiked on hype. AI 3x off lows. "
            "Subsequently gave most back. Classic hype-and-dump cycle."
        ),
        expected_difficulty="hard",
    ),

    # ============================================================
    # POST-EVENT
    # ============================================================

    MarketScenario(
        name="post_election_nov_2016",
        category="post_event",
        symbols=["SPY", "TLT", "XLF", "XLE"],
        start=_d(2016, 11, 7), end=_d(2016, 12, 30),
        description=(
            "Trump election reaction: SPY +5%, TLT -8%, XLF +12%, XLE +7%. "
            "Reflation trade. Sector rotation in days was the opportunity."
        ),
        expected_difficulty="moderate",
    ),
    MarketScenario(
        name="post_election_nov_2020",
        category="post_event",
        symbols=["SPY", "QQQ", "TSLA", "XLE", "TLT"],
        start=_d(2020, 11, 3), end=_d(2021, 1, 15),
        description=(
            "Biden election + vaccine news (Nov 9). XLE +40% in 6 weeks on reopening. "
            "TSLA added to S&P 500. Rotation from growth to value."
        ),
        expected_difficulty="moderate",
    ),
    MarketScenario(
        name="post_fomc_oct_2018",
        category="post_event",
        symbols=["SPY", "TLT", "QQQ", "IWM"],
        start=_d(2018, 10, 3), end=_d(2018, 10, 31),
        description=(
            "Powell 'long way from neutral' speech. SPY -7% in 6 days. "
            "Accelerated the Q4 2018 selloff."
        ),
        expected_difficulty="hard",
    ),
    MarketScenario(
        name="post_fomc_rate_pivot_2022",
        category="post_event",
        symbols=["SPY", "TLT", "QQQ", "XLU"],
        start=_d(2022, 11, 1), end=_d(2023, 1, 31),
        description=(
            "Fed pivot hints + CPI miss Oct 2022. SPY +14% in 6 weeks. "
            "Best short-covering rally of 2022 before re-test in Q1 2023."
        ),
        expected_difficulty="hard",
    ),

    # ============================================================
    # SECTOR
    # ============================================================

    MarketScenario(
        name="energy_supercycle_2021",
        category="sector",
        symbols=["XLE", "XOM", "CVX", "SPY", "USO"],
        start=_d(2020, 11, 1), end=_d(2022, 6, 30),
        description=(
            "Energy sector supercycle. XLE +120% over 20 months from COVID lows. "
            "Reopening + supply discipline + Ukraine war. The only sector that worked in 2022."
        ),
        expected_difficulty="easy",
    ),
    MarketScenario(
        name="tech_deflation_2022",
        category="sector",
        symbols=["QQQ", "META", "NFLX", "PYPL", "SNAP"],
        start=_d(2022, 1, 1), end=_d(2022, 10, 31),
        description=(
            "Rising rates collapse valuation multiples. META -72%, NFLX -74%, PYPL -75%. "
            "Growth-at-any-price era ends. Worst year for tech since dot-com."
        ),
        expected_difficulty="brutal",
    ),
    MarketScenario(
        name="bank_crisis_cycle_2023",
        category="sector",
        symbols=["XLF", "KRE", "JPM", "BAC", "GLD"],
        start=_d(2023, 2, 1), end=_d(2023, 6, 30),
        description=(
            "Regional bank stress cycle: SVB → Signature → First Republic → contagion fear. "
            "KRE -40% peak to trough. JPM +6% (flight to quality large-cap). GLD benefited."
        ),
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
    total_symbols = sum(len(s.symbols) for s in SCENARIOS)
    lines = [
        f"Scenario library: {len(SCENARIOS)} episodes across {len(by_cat)} categories",
        f"Total symbol-runs:  {total_symbols}",
        "",
    ]
    for cat in sorted(by_cat):
        lines.append(f"  {cat.upper()}  ({len(by_cat[cat])} scenarios)")
        for s in by_cat[cat]:
            lines.append(
                f"    • {s.name:<35}  {s.start.date()} → {s.end.date()}  "
                f"({s.days:>4}d)  [{s.expected_difficulty}]  "
                f"syms={','.join(s.symbols)}"
            )
    return "\n".join(lines)
