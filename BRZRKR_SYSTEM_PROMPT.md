# BRZRKR — Full System Brief

Use this document to understand, critique, and improve the BRZRKR trading system. Everything below is a faithful description of what exists in the codebase today.

---

## What BRZRKR Is

BRZRKR is a fully autonomous AI trading system running on a local macOS machine. It watches the stock market during trading hours, generates high-confidence trade signals using a multi-model AI ensemble, sizes and places bracket orders through Alpaca's paper (and optionally live) trading API, and learns from every trade it closes. The entire system is controlled through a custom gothic-styled desktop app written in Python.

The aesthetic is deliberately heavy — blood-red on near-black, gothic glyphs, Berserk/Vagabond manga typography. This is not decoration; it's a design philosophy: the system should feel like a serious, dangerous instrument.

---

## The Desktop App (brzrkr_app/)

The app is built with CustomTkinter (Python's modern Tkinter wrapper). It runs as a native macOS window at 1440×900, dark mode, no browser, no Electron.

**Navigation — eight pages in a left sidebar:**

- **Status** — Live broker snapshot: account equity, open positions, daily P&L, agent running/stopped, API key health matrix, recent signal files, live log tail. Start/Stop agent buttons with options.
- **Trades** — Open positions and recent orders with color-coded P&L (green/red, dims on tiny gains).
- **Console** — Manual bracket order form. Per-position "Seal" buttons to close individual trades. "Banish ALL" to close everything at once. Per-order cancel buttons.
- **Market Intel** — Custom-drawn candlestick charts (pure Tkinter Canvas, no matplotlib). Live watchlist. "Power Plays" (highest-confidence signals). Scanner output.
- **Backtests (Proving Grounds)** — Historical scenario battery results, four live simulation cards showing active backtests in real time, win rate / Sharpe / drawdown stats, per-category breakdown.
- **Strategy** — Read-only view of the signal pipeline logic and current model weights.
- **Postmortem** — Browse the failure-mode knowledge base (SQLite). Searchable codex of quant failure patterns the system has learned or been seeded with.
- **Admin** — File registry of every script, live activity pulse bars (log writes / signal writes / order writes / lessons learned), "Inscribe" export buttons.
- **System** — Process health, disk usage, data freshness indicators.

A background `BrokerPoller` thread queries Alpaca every 5 seconds and pushes snapshots into a Queue. The Tk event loop drains the Queue and refreshes every visible page without blocking the UI.

---

## The AI Signal Pipeline

### Step 1 — Universe Scanning

The agent maintains a watchlist of tickers. On each 5-minute tick during market hours it fetches the latest OHLCV bars for each ticker via Alpaca's market data API (with yfinance as fallback for historical data).

### Step 2 — Feature Engineering (src/features/feature_engineer.py)

Raw price/volume data is transformed into a feature bundle containing:
- Price momentum across multiple timeframes (5d, 10d, 20d, 50d)
- Volatility metrics (realized vol, ATR, Bollinger bandwidth)
- Volume profile and relative volume
- EMA slopes (20, 50, 200)
- RSI, MACD, Stochastic
- Sentiment scores (FinBERT-encoded news headlines when available)
- Macro context (VIX level, sector rotation signals)

### Step 3 — The Ensemble (src/model/ensemble.py)

Five sub-models each vote on direction (long/short) and confidence (0.0–1.0):

**1. MultiFactorMomentum** (label: "lstm") — Weight: 22%
Multi-lag momentum model. Combines 5d/10d/20d/50d returns with volume confirmation and trend quality R². The volatility-adjusted edge (Sharpe-like score) maps to confidence. Strong trend + high-volume confirmation can reach 0.88+ confidence.

**2. TabularSentiment/XGBPredictor** (label: "xgboost") — Weight: 28%
Trained XGBoost model using 14 tabular features. If a trained model file exists at `models/xgb.json` it loads it; otherwise falls back to a heuristic. Highest weight because XGBoost generalizes well on tabular data.

**3. RegimeAwareModel** (label: "transformer") — Weight: 20%
Classifies the current market regime using ADX, Bollinger bandwidth, realized vol percentile, and EMA slopes. In trending regimes (ADX > 25): follows trend direction, confidence scales with ADX strength and triple-alignment bonus (ADX direction + price above EMA20 + EMA20 above EMA50 all agree → 1.20× multiplier). Ceiling: 0.93. In ranging regimes: mean-reversion at Bollinger extremes. In volatile regimes: reduced conviction.

**4. TechnicalConfluenceAgent** (label: "confluence") — Weight: 20%
Eight independent technical indicators each vote -1 (bearish), 0 (neutral), or +1 (bullish): RSI relative to 50, MACD histogram sign, price vs EMA-20, EMA-20 vs EMA-50, Bollinger position, relative volume, ATR momentum, Stochastic. Score = sum of votes. Formula: `confidence = 0.50 + |score| * 0.048`. Six of eight agreeing = 0.86 confidence. Seven of eight = 0.92. This is the most reliable path to high-confidence signals.

**5. BreakoutDetector** (label: "breakout") — Weight: 10%
Checks five quality conditions for a genuine breakout: (1) price clears a 20-bar high/low by at least 0.25× ATR, (2) volume > 1.25× 20-bar average, (3) ATR is expanding (current vs prior 10-bar), (4) EMA alignment (EMA20 > EMA50 for longs), (5) clean close (last candle closes near the high for longs). Four of five conditions met = 0.82 confidence. Five of five = 0.90+.

**6. RL Agent** (label: "rl") — Weight: 0% currently
A PPO reinforcement learning agent backed by a custom gym environment. Loads if a trained checkpoint exists. Currently untrained and inactive.

**Ensemble aggregation:**
1. Weights are normalized across active models.
2. Each model's vote is `(2 × confidence − 1) × direction_sign` — a confidence-weighted directional score centered at zero (neutral = 0, maximum bullish = +1, maximum bearish = -1). This prevents low-confidence models from polluting the direction with a 0.5 score.
3. Weighted mean of individual confidences is the base confidence.
4. Agreement factor: `|directional_vote| / max_possible_vote` — how aligned the models are.
5. Final confidence = `base_conf × max(0.60, agreement)`.

### Step 4 — Signal Calibration (src/learning/signal_calibrator.py)

Raw confidence is post-processed by a calibrator that reads the closed trade journal and buckets historical win rates in 0.05-wide confidence bins. If signals with 0.52 raw confidence have historically won 67% of the time (above the expected 50%), that bucket gets scaled up. Requires at least 5 closed trades per bucket before trusting calibration. Uses an isotonic-inspired formula: `calibrated = raw × (1 + (actual_win_rate − 0.50) × 2 × 0.40)`.

### Step 5 — Signal Generation & Risk Filtering (src/signals/signal_generator.py)

A signal is only emitted if calibrated confidence ≥ 0.55. Below that: silence, no trade.

Every signal then passes through the risk manager:
- **ATR-based stop loss**: entry ± 2.0× ATR(14).
- **Take profit**: stop distance × 1.5 reward/risk ratio (configurable up to 2.0×).
- **Volatility filter**: if realized vol > 4% daily (~63% annualized) or VIX > 35 → no trade.
- **Correlation filter**: if the proposed trade has > 0.70 correlation with an existing open position → skip (prevents doubling up on the same bet).
- **Blackout window**: no new entries within 15 minutes of market close.

### Step 6 — Position Sizing (src/risk/risk_manager.py)

Priority 1 — **Kelly half-criterion** (if ≥ 30 closed trades with stats):
`f* = (p × b − q) / b × 0.5` where p = win rate, q = 1−p, b = avg_win / avg_loss. Produces a fraction of equity to risk. Hard cap at 6% of account equity per position.

Priority 2 — **Fixed-fraction fallback** (when trade history is thin):
`equity × 1% / stop_distance_pct`, scaled by confidence between 0.5× and 1.0×.

Maximum simultaneous positions: 8. Maximum daily drawdown before halt: 3%.

### Step 7 — Order Execution (src/execution/broker.py)

AlpacaExecutor submits bracket orders (entry + stop-loss + take-profit in one atomic order) via Alpaca's paper trading API by default. Real money requires `ALPACA_LIVE=true` in environment AND `live_money=True` in code. Fractional shares are not used. Orders are idempotent within a session (will not re-submit a client_order_id it already placed).

---

## The Learning Loop

### Trade Journal (src/learning/trade_journal.py)
Every signal, entry, and exit is logged to `data/trade_journal.jsonl`. Fields: asset, direction, entry_price, exit_price, confidence, pnl_usd, status (open/closed), ensemble components.

### Session Observer (src/learning/observer.py)
After each trading session, SessionObserver reviews what happened: trades missed, stops hit, positions that reversed after exit. Writes new Lessons to the PostmortemDB when patterns are detected.

### PostmortemDB (src/learning/postmortem_db.py)
SQLite knowledge base (`data/brzrkr.db`) of quant failure patterns. Pre-seeded with 50+ lessons from quant history (momentum crashes, carry trade blowups, earnings gaps, liquidity crises). Self-updates as the system encounters new failure modes.

### Online Learner (src/learning/online_learner.py)
Incremental model updates using closed trade outcomes. Adjusts ensemble weights based on which sub-models were most predictive for recent closed trades.

### Signal Calibrator (src/learning/signal_calibrator.py)
Described above. Updates every session using the closed trade journal.

---

## Backtesting System

### Scenarios (src/backtest/scenarios.py)
40 hand-curated historical market episodes across six categories:
- **crash**: dot_com_crash_2000, covid_crash_2020, dec_2018_selloff, aug_2015_china_crash
- **crisis**: great_recession_2007, sept_2008_lehman, euro_debt_crisis_2012, svb_collapse_2023
- **vol_spike**: flash_crash_2010, volmageddon_feb_2018, covid_vol_march_2020, taper_tantrum_2013, brexit_vote_2016
- **rally**: post_crisis_rally_2009, and others across 2013–2024
- **sideways**: choppy/low-volatility periods
- **rate_shock**: taper tantrum, rate hike cycles

Each scenario specifies a date range, a list of symbols, a difficulty rating (mild / moderate / severe / brutal), and a description.

### Scenario Runner (src/backtest/scenario_runner.py)
Runs each scenario × symbol combination through the full signal pipeline (same ensemble, same risk manager, same confidence threshold as live trading). Runs 4 slots in parallel. Each slot streams live status to a JSON file every bar, which the desktop app polls every 400ms to animate the live sim cards.

**Data fetching hierarchy (with 20/25/15 second timeouts):**
1. Alpaca market data API (fast, covers ~5 years)
2. yfinance (handles Yahoo anti-bot auth, reliable for 2000-era data)
3. Yahoo Finance direct scraper (last resort)

### Results (data/scenario_runs/)
- `_results_so_far.csv` — running partial file for the active batch
- `scenario_report_YYYYMMDDTHHMMSSZ.csv` — completed batch archives
- `_live.json`, `_live_1.json`, `_live_2.json`, `_live_3.json` — real-time slot status

### Continuous Practice (continuous_practice.py)
Runs the 40-scenario battery in an infinite loop (with configurable rest between batches). Archives completed batches to `data/archive/`. Stops on `CONTINUOUS_STOP` sentinel file or Ctrl-C. Started/stopped from the Backtests tab.

---

## Data Sources

| Source | What it provides | Key files |
|--------|-----------------|-----------|
| Alpaca Market Data API | Real-time + historical OHLCV bars | src/data_loader.py |
| Yahoo Finance (scraper) | Historical OHLCV fallback | src/data_scraper.py |
| yfinance library | Historical OHLCV, options chains | src/data_alternatives.py |
| Alpaca Trading API | Order submission, portfolio state | src/execution/broker.py |
| FinBERT / news RSS | Sentiment scoring on headlines | src/model/sentiment_encoder.py |
| FRED (Federal Reserve) | Macro indicators | src/data_loader.py |
| Polygon.io | Options flow, alternative bars | src/data_loader.py |

All credentials live in `.env` (never committed). Loaded via `python-dotenv`.

---

## Key File Map

```
BRZRKR trader/
│
├── BRZRKR.py                     ← launch the desktop app
├── agent.py                      ← launch the trading agent (CLI)
├── auto_trader.py                ← fully automated mode (agent + scanning)
├── continuous_practice.py        ← run scenario battery forever
├── weekend_practice.py           ← single batch scenario run
├── train.py                      ← train XGBoost on feature dataset
├── train_lstm.py                 ← train LSTM
├── train_rl.py                   ← train RL agent
│
├── brzrkr_app/                   ← desktop app package
│   ├── main_window.py            ← sidebar, page stack, broker poller
│   ├── theme.py                  ← colors, fonts, glyphs (C.BLOOD, G.RUNE_*)
│   ├── widgets.py                ← reusable UI components (LiveSimCard, EquityCurve, etc.)
│   ├── poller.py                 ← background broker polling thread
│   └── pages/
│       ├── status.py             ← agent control, equity, positions
│       ├── trades.py             ← open positions, orders
│       ├── market.py             ← candlestick charts, watchlist
│       ├── backtests.py          ← scenario battery UI, live sim cards
│       ├── strategy.py           ← signal pipeline viewer
│       ├── postmortem.py         ← failure codex browser
│       ├── admin.py              ← file registry, export tools
│       └── system.py             ← process health
│
├── src/
│   ├── agent/trading_agent.py    ← main trading loop (data→features→predict→trade)
│   ├── model/
│   │   ├── ensemble.py           ← EnsemblePredictor, EnsembleWeights
│   │   ├── momentum_model.py     ← MultiFactorMomentum (LSTM slot)
│   │   ├── regime_model.py       ← RegimeAwareModel (Transformer slot)
│   │   ├── xgb_predictor.py      ← XGBoost predictor
│   │   ├── technical_confluence.py ← 8-indicator voting model
│   │   └── breakout_detector.py  ← 5-condition breakout model
│   ├── signals/signal_generator.py ← confidence gate + signal JSON builder
│   ├── risk/risk_manager.py      ← sizing, stops, filters
│   ├── execution/broker.py       ← Alpaca bracket order submission
│   ├── features/feature_engineer.py ← OHLCV → feature bundle
│   ├── backtest/
│   │   ├── scenario_runner.py    ← runs full ensemble on historical episodes
│   │   ├── backtest_runner.py    ← bar-by-bar backtester
│   │   ├── scenarios.py          ← 40 named market scenarios
│   │   └── live_status.py        ← atomic JSON status writer for UI
│   └── learning/
│       ├── postmortem_db.py      ← SQLite failure-mode knowledge base
│       ├── signal_calibrator.py  ← isotonic confidence adjustment
│       ├── observer.py           ← session-level self-reflection
│       ├── trade_journal.py      ← JSONL trade log
│       └── online_learner.py     ← incremental model weight updates
│
├── data/
│   ├── signals/                  ← live signal JSON files (one per active trade)
│   ├── scenario_runs/            ← backtest results CSV + live status JSON
│   ├── trade_journal.jsonl       ← every trade entry/exit ever made
│   └── brzrkr.db                 ← SQLite: postmortem lessons + sessions
│
└── models/
    ├── xgb.json                  ← trained XGBoost model
    ├── lstm_*.pt                 ← trained LSTM checkpoints (if trained)
    └── rl_agent/                 ← PPO checkpoint (if trained)
```

---

## Current Thresholds and Constants

| Parameter | Value | Location |
|-----------|-------|----------|
| Confidence threshold (live) | 0.55 | AgentConfig, signal_generator.py |
| Confidence threshold (backtest) | 0.75 | ScenarioRunner |
| Max positions | 8 | AgentConfig |
| Max daily loss before halt | 3% | AgentConfig |
| Max position size | 6% of equity | risk_manager.py |
| Base risk per trade | 1% of equity | risk_manager.py |
| Kelly fraction | 0.5 (half-Kelly) | risk_manager.py |
| ATR stop multiplier | 2.0× | risk_manager.py |
| Reward/risk target | 1.5:1 (→ 2.0:1) | risk_manager.py |
| Correlation block | 0.70 | risk_manager.py |
| VIX crisis level | 35 | risk_manager.py |
| Pre-close blackout | 15 min | AgentConfig |
| Tick interval | 5 min | AgentConfig |

---

## Known Weaknesses / Open Problems

1. **Backtest confidence threshold mismatch** — the live agent uses 0.55 but the backtest scenario runner uses 0.75. This means backtest results represent a stricter strategy than what actually runs live. The thresholds should be unified.

2. **RL agent is untrained and inactive** — the PPO agent exists in code but has never been trained. It gets 0% weight in the ensemble.

3. **No intraday data** — the system operates on daily bars. All models, features, and scenarios are built around end-of-day OHLCV. Adding 1-hour or 15-minute bars would significantly improve entry timing.

4. **Signal calibrator cold-start** — the calibrator only adjusts confidence after ≥5 closed trades per bucket. A fresh deployment has no calibration data and runs on raw model confidence for weeks.

5. **No sector awareness in the ensemble** — none of the five sub-models receives sector ETF flows, rotation signals, or relative strength between sectors. The system trades individual symbols without knowing if money is flowing out of tech into energy, for example.

6. **XGBoost model was trained on limited data** — the trained XGB model uses 14 features on whatever training set was available. It may be overfit to recent regimes. Retraining on diverse historical periods is needed.

7. **No earnings / event blackout** — the system does not check if an earnings announcement or major economic release (CPI, FOMC) is within 48 hours of a planned entry. Entering before a binary event is a known failure mode that is in the PostmortemDB but not enforced in code.

8. **Paper trading only** — all positions in the journal are paper trades. The system has never traded real money. Slippage, partial fills, and liquidity constraints are not modeled.

9. **No options trading** — despite the original design mentioning options/IV, the signal generator and broker executor operate entirely with equity (stock) positions. The options scanner code exists but is not wired into the main agent loop.

10. **Dual SQQQ/TQQQ positions** — the correlation filter only checks `r > 0.70` between tickers, but does not check if two positions are conceptually inverse (e.g., being long TQQQ and short SQQQ is redundant exposure to the same underlying). A semantic position check is missing.

---

## How the Pieces Connect (end-to-end flow)

```
Market Hours Tick (every 5 min)
        │
        ▼
 [BrokerPoller] ──► Alpaca: get portfolio, positions, orders
        │
        ▼
 [TradingAgent._evaluate_symbol(ticker)]
        │
        ├─► fetch OHLCV bars (Alpaca → yfinance fallback)
        │
        ├─► FeatureEngineer.build_features(bundle)
        │        └─ momentum, vol, sentiment, macro context
        │
        ├─► EnsemblePredictor.predict(features)
        │        ├─ MultiFactorMomentum     (22%)
        │        ├─ XGBPredictor            (28%)
        │        ├─ RegimeAwareModel        (20%)
        │        ├─ TechnicalConfluenceAgent(20%)
        │        └─ BreakoutDetector        (10%)
        │
        ├─► SignalCalibrator.calibrate(raw_confidence)
        │
        ├─► confidence < 0.55? → SKIP
        │
        ├─► RiskManager: ATR stop, TP, vol filter, correlation, blackout
        │
        ├─► calculate_position_size (Kelly or fixed-fraction)
        │
        ├─► generate_signal() → write JSON to data/signals/
        │
        └─► AlpacaExecutor.submit_bracket_order()
                 └─ entry + stop-loss + take-profit in one order

Closed trade → TradeJournal.log_exit()
                    │
                    ├─► SignalCalibrator updates win-rate buckets
                    ├─► SessionObserver checks for failure patterns
                    └─► PostmortemDB.add_lesson() if pattern matched
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Desktop UI | CustomTkinter (Tkinter wrapper) |
| Charts | Pure Tkinter Canvas (custom candlestick renderer) |
| ML models | XGBoost, PyTorch (LSTM/Transformer stubs), scikit-learn |
| RL | Stable-Baselines3 PPO, custom gym environment |
| Broker API | alpaca-py (Alpaca Trading + Market Data) |
| Historical data | Alpaca + yfinance |
| Database | SQLite (via Python sqlite3) |
| Sentiment | Transformers (FinBERT) |
| Config | python-dotenv (.env file) |
| Testing | pytest |

---

## What "Improving" This System Means

If you are an AI being asked to improve BRZRKR, here are the highest-leverage areas ranked by expected impact:

1. **Unify the confidence threshold** — backtest uses 0.75, live uses 0.55. Either raise live to 0.75 (fewer but higher-quality signals) or lower backtest to 0.55 (more realistic simulation). The gap makes backtests unrepresentative.

2. **Add an earnings/event blackout** — check whether any position entry falls within 48h of an earnings announcement or FOMC release. This is the #1 preventable failure mode in the PostmortemDB.

3. **Train the RL agent** — the gym environment (`src/rl/trading_env.py`) is already built. Training PPO on the historical scenario data would give the ensemble a genuine sixth vote.

4. **Add intraday bars** — the feature engineer and scenario runner both accept arbitrary bar intervals. Switching from daily to 1-hour bars for entry timing (while keeping daily for trend/regime detection) would dramatically improve fill quality.

5. **Wire in sector rotation** — add a sector ETF relative strength signal (XLK vs XLE vs XLF vs XLY) to the feature bundle. RegimeAwareModel could incorporate this to avoid fighting sector headwinds.

6. **Retrain XGBoost** — run `train.py` on the scenario backtest data from `data/scenario_runs/` to retrain XGB with diverse historical regimes, not just recent live data.

7. **Fix the semantic correlation check** — add inverse-ETF awareness to the correlation filter. TQQQ/SQQQ, SPXU/SPXL, TZA/TNA should block each other regardless of measured correlation coefficient.

8. **Calibrate with more closed trades** — the calibrator only activates after 5 trades per bucket, which takes weeks of live trading. Bootstrap it by replaying the backtest trade-by-trade and building an initial calibration map from historical scenario results.
