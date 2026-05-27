# ✠ BRZRKR ✠

> *The forge of trade — gothic dark dashboard atop a production-ready quant scaffold.*

Production-ready scaffold for an AI options/futures enhancer that
predicts directional moves and volatility, runs an ensemble of models
(LSTM + XGBoost + Transformer) over FinBERT-scored sentiment + macro
context, and emits broker-ready trade signals through a strict risk
manager (Kelly half-sizing, ATR stops, correlation/volatility/blackout
filters, drawdown monitoring). The visible face is **BRZRKR**, a
native macOS desktop app with six tabs themed after Berserk / Shamo /
Vagabond — blood crimson on iron black, custom-drawn candlestick
charts, ornate gothic glyphs.

## ✠ BRZRKR Desktop App

Six tabs, all live-updating from a background broker poller:

| Tab | What |
| --- | --- |
| **❰ STATUS ❱** | Broker / equity / open positions; Agent control panel (Start/Stop with options); API key matrix; recent signals; log tail. |
| **❰ TRADES ❱** | Open positions + recent orders with color-coded P&L. |
| **❰ CONSOLE ❱** | Manual order form (bracket), per-position *Seal* buttons, per-order *Banish* buttons, *Banish ALL*. |
| **❰ MARKET ❱** | Custom-drawn candlestick charts (pure tk Canvas, ink-brushed, no matplotlib). |
| **❰ POSTMORTEM ❱** | Browse / filter the failure-mode codex, run the correlation analyzer. |
| **❰ ADMIN ❱** | File registry of every script in the project, live activity pulse bars (log / signals / orders / lessons), *Inscribe* buttons that export system reports + project tree + postmortem MD. |

**Run it (after `./setup.sh` once):**

```bash
source venv/bin/activate
python BRZRKR.py
```

**Drop on the Dock (one-click launch):**

```bash
open .
# drag BRZRKR.command from Finder onto the Dock
```

**Build a real .app bundle:**

```bash
./build_brzrkr.sh        # produces dist/BRZRKR.app
open dist/BRZRKR.app
# install:
mv dist/BRZRKR.app /Applications/
```

The `.app` bundle is a thin launcher — it runs `BRZRKR.py` from the
project source on every launch, so any code update is reflected
immediately. **You do not need to rebuild the bundle when you update
Python files.** Only rebuild when adding new top-level Python
dependencies.

A custom icon is generated automatically by `brzrkr_app/icon.py` —
circular black emblem with a stylised "B" in crimson, sword-slash in
ember, ornate gothic ring. Saved to `assets/BRZRKR_icon.png`.

## ✠ A note on the naming

The internal Python modules under `src/` keep their original
snake_case names. This is intentional: prefixing every module with
`BRZRKR_` would break 30+ existing imports and the muscle memory for
every command. **Branding lives at the top level** (`BRZRKR.py`
launcher, the `brzrkr_app/` package, this README, the icon, the .app
bundle). The internal engine kept its working name so it keeps
working.

The old browser dashboard (`dashboard.py`) and the previous
CustomTkinter desktop app (`desktop_app.py`) are archived to
`_legacy/` — nothing important lost, easily recoverable, deletable
with `rm -rf _legacy/` once you're sure you won't need them.

---



## Folder layout

```
trading_enhancer/
├── data/                       # raw + processed market data (gitignored)
├── models/                     # trained checkpoints & serialized models
├── src/
│   ├── data_loader.py          # OHLCV, options, news, macro fetchers
│   ├── model/
│   │   ├── transformer_backbone.py   # 3‑layer Transformer, seq_len=256
│   │   ├── sentiment_encoder.py      # FinBERT wrapper
│   │   ├── volatility_module.py      # GARCH(1,1) + IV spline surface
│   │   └── ensemble.py               # weighted LSTM/XGB/Transformer
│   ├── risk/
│   │   └── risk_manager.py     # Kelly, stops, filters, drawdown
│   ├── signals/
│   │   └── signal_generator.py # JSON signal, ≥75% confidence
│   ├── backtest/
│   │   └── backtest_runner.py  # walk‑forward backtester
│   └── utils/
│       ├── logging_setup.py    # rotating file + console logging
│       └── cron_job.sh         # daily 02:00 UTC scheduler
├── config/
│   ├── config.yaml             # all non‑secret parameters
│   └── secrets.yaml            # template — DO NOT COMMIT real keys
├── tests/                      # pytest suite
├── notebooks/                  # exploratory analysis / prototyping
├── docs/                       # architecture & deployment guides
├── requirements.txt
├── setup.sh
└── run.py
```

## How to use

Every step below is a copy‑paste block. Run them top to bottom the
first time. Assumes macOS / Linux + Python ≥ 3.10. On Windows use
WSL2 or Git Bash — the `.sh` scripts and `source` won't run in PowerShell.

### Step 0 — Check prerequisites

Run these one at a time:

```bash
python3 --version
```

```bash
git --version
```

`python3 --version` must print 3.10 or newer. If it's older, install a
newer version:

- macOS: `brew install python@3.11`
- Ubuntu: `sudo apt install python3.11 python3.11-venv`

### Step 1 — Enter the project and create the virtual environment

⚠️ **Important:** zsh on macOS does not treat `#` as an inline comment in
interactive mode. Paste each block exactly as shown — never paste a
line and its trailing `# explanation`. Every block below is already
clean.

Move into the project root:

```bash
cd /Users/max51/trading_enhancer
```

Mark the bootstrap script executable (only needed once):

```bash
chmod +x setup.sh
```

Run the bootstrapper. It creates `./venv`, installs everything in
`requirements.txt`, then imports every module to confirm the install
worked. You should see `Environment ready` at the end:

```bash
./setup.sh
```

Activate the venv for the current shell — your prompt should now start
with `(venv)`. Re‑run this command every time you open a new terminal:

```bash
source venv/bin/activate
```

To deactivate later: `deactivate`.

### Step 2 — Add your API keys

Pick **one** of the two options below. Option A (env vars) is the
simplest; Option B (`.env` file) is best for cron.

**Option A — paste keys into your shell:**

```bash
export ALPACA_API_KEY="paste-your-key-here"
export ALPACA_SECRET_KEY="paste-your-secret-here"
export TRADIER_API_KEY="paste-your-key-here"
export UNUSUAL_WHALES_API_KEY="paste-your-key-here"
export NEWSAPI_KEY="paste-your-key-here"
export FRED_API_KEY="paste-your-key-here"
export HF_TOKEN="paste-your-key-here"
```

(The last one, `HF_TOKEN`, is optional — only needed for gated FinBERT
variants on Hugging Face.)

(These only last for the current terminal session. To persist them,
add the lines to `~/.zshrc` or `~/.bashrc`.)

**Option B — create a `.env` file (loaded by `cron_job.sh`):**

Paste the whole block at once (the `EOF` marker tells the shell where
the file ends):

```bash
cat > .env <<'EOF'
ALPACA_API_KEY=paste-your-key-here
ALPACA_SECRET_KEY=paste-your-secret-here
TRADIER_API_KEY=paste-your-key-here
UNUSUAL_WHALES_API_KEY=paste-your-key-here
NEWSAPI_KEY=paste-your-key-here
FRED_API_KEY=paste-your-key-here
HF_TOKEN=
EOF
```

Then lock the file down so only your user can read it:

```bash
chmod 600 .env
```

Verify the keys are visible to Python:

```bash
python -c "import os; print('ALPACA OK' if os.getenv('ALPACA_API_KEY') else 'MISSING')"
```

Where to get keys:

| Vendor          | Sign up                                                |
| --------------- | ------------------------------------------------------ |
| Alpaca          | https://app.alpaca.markets (paper trading is free)     |
| Tradier         | https://developer.tradier.com                          |
| Unusual Whales  | https://unusualwhales.com                              |
| NewsAPI         | https://newsapi.org                                    |
| FRED            | https://fred.stlouisfed.org/docs/api/api_key.html      |
| Polygon (opt.)  | https://polygon.io                                     |
| Hugging Face    | https://huggingface.co/settings/tokens                 |

### Step 3 — Tune non‑secret parameters (optional)

Open [`config/config.yaml`](config/config.yaml) in any editor and
adjust:

- `data.universe` — the symbols you want to trade.
- `signals.confidence_threshold` — default `0.75`. Raise to be picker.
- `risk.max_position_pct` — default `0.05` (5 % of equity per trade).
- `live.initial_account_equity_usd` — sizing baseline.

You don't need to touch this for a first run.

### Step 4 — Run the test suite (sanity check)

```bash
pytest -q
```

Expected output: every test passes, finishing in a few seconds. If
something fails, fix the environment before going further — the live
pipeline uses the same code paths.

### Step 5 — Run a backtest on sample data

```bash
# 1. Put a CSV with columns: timestamp, open, high, low, close, volume
#    at data/sample.csv. (You can generate one from yfinance — see below.)
python -c "
import yfinance as yf
df = yf.download('SPY', period='2y', interval='1d', auto_adjust=False)
df = df.rename(columns=str.lower).reset_index().rename(columns={'date':'timestamp'})
df.to_csv('data/sample.csv', index=False)
print('Wrote', len(df), 'rows to data/sample.csv')
"

# 2. Run the backtest.
python run.py --mode backtest --data data/sample.csv
```

Expected output: a JSON summary on stdout with `trades`, `win_rate`,
`avg_pnl`, `sharpe`, `max_drawdown_pct`, `final_equity`. Detailed
artifacts land in:

- `data/backtest_out/equity_curve.csv`
- `data/backtest_out/trades.csv`

### Step 6 — Run the live pipeline once

By default this only **generates signals** (writes JSON files); it
does not place any orders. Add `--execute` to actually submit bracket
orders to Alpaca (paper by default — fake money, real order
lifecycle).

**Signal‑only run (safe, no orders placed):**

```bash
python run.py --mode live
```

For each symbol in `data.universe`, this fetches data, runs the
ensemble, applies risk filters, and — if confidence ≥ 75 % — writes a
JSON signal to `data/signals/<symbol>_<timestamp>.json`. No signal
files means no symbol cleared the threshold this run (normal).

Inspect a signal:

```bash
ls data/signals/
cat data/signals/$(ls -t data/signals/ | head -n1)
```

**Paper trading (real Alpaca paper account, fake money):**

```bash
python run.py --mode live --execute
```

This pulls your real Alpaca paper account equity, sizes positions
against it, and submits bracket orders (entry + stop + take‑profit
in one shot). Watch the orders fill on
https://app.alpaca.markets/paper/dashboard/overview.

**Real money (use only after weeks of paper success):**

```bash
export ALPACA_LIVE=true
python run.py --mode live --execute --live-money
```

The `--live-money` flag is rejected unless `ALPACA_LIVE=true` is also
set in the environment — both guards must be true to route real
orders. Even then, start with a single low‑volatility symbol in
`data.universe` and watch every fill.

### Data source modes (`--source`)

The pipeline can pull from three different stacks:

| Flag | Where data comes from | API keys needed |
| --- | --- | --- |
| `--source api` (default) | Alpaca + Tradier + Finnhub + NewsAPI + FRED + Polygon | Each vendor's key (most are free tiers) |
| `--source scraper` | Yahoo + SEC EDGAR + Treasury + CFTC + FINRA + Reddit + Google News + Congress + StockTwits + HN + Wikipedia | Alpaca only (execution) |
| `--source both` | API primary (canonical OHLCV) + scraper extras merged in | Alpaca + whatever API keys you have |

**Scraper-only run** (zero secondary API keys, all 13 sources):

```bash
python run.py --mode live --source scraper
```

**With FinBERT sentiment scoring over every text source** (adds ~30s for first model load):

```bash
python run.py --mode live --source scraper --score-sentiment
```

**API primary + scraper extras (Reddit, Congress, Wikipedia attention) layered on top:**

```bash
python run.py --mode live --source both --execute
```

Every run goes through the same `FeatureEngineer` (`src/features/feature_engineer.py`),
which turns the raw data bundle into:

- A **price DataFrame** with technical indicators (returns, SMA cross,
  RSI, realized vol, volume z-score)
- A **context dict** of point-in-time scalars (sentiment aggregates per
  source, congress net buys, insider filing count, P/C ratio, IV ATM,
  yield curve spread, VIX level, Wikipedia pageview z-score)

These are passed to the ensemble's three sub-models (`_PriceMomentum`,
`_TabularSentiment`, `_TransformerVol`) which all consume both layers.
Each emitted signal JSON now includes a `context_features` block
recording exactly which features drove the prediction.

### Step 7 — Schedule the daily 02:00 UTC run (optional)

```bash
# 1. Tell the cron wrapper where the project lives.
sed -i.bak "s|/path/to/trading_enhancer|$(pwd)|" src/utils/cron_job.sh
chmod +x src/utils/cron_job.sh

# 2. Install the cron entry.
( crontab -l 2>/dev/null; echo "0 2 * * * $(pwd)/src/utils/cron_job.sh" ) | crontab -

# 3. Confirm it's there.
crontab -l | grep trading_enhancer
```

Logs land in:

- `trading_enhancer.log` — application log (rotated, 10 MB × 5).
- `cron.log` — captured stdout/stderr from each cron run.

Remove the entry later with `crontab -e` (delete the line) or
`crontab -r` (clear all entries — be careful).

### Daily workflow after first‑time setup

```bash
cd /Users/max51/trading_enhancer
source venv/bin/activate
python run.py --mode live          # or --mode backtest --data <file>
deactivate
```

### Autonomous trading agent (`agent.py`)

`agent.py` is the always-on cousin of `run.py`. It loops continuously
during market hours, evaluates every symbol on each tick, and places
bracket orders through `AlpacaExecutor` — with portfolio-level
guardrails on top of the per-trade risk manager.

**Safest first run** (paper, dry-run — logs would-be trades, never submits):

```bash
python agent.py --dry-run
```

**Paper-money autonomous trading**:

```bash
python agent.py --execute
```

**With richer data sources**:

```bash
python agent.py --execute --source both --score-sentiment
```

**Stop cleanly** with either approach:

- `Ctrl-C` — finishes the current tick, logs a session summary, exits.
- `touch AGENT_STOP` — picked up at the top of the next tick. Works
  even when the agent is running detached (e.g. via `nohup`, `tmux`,
  `screen`, or systemd).

**Tuning flags** (all optional):

| Flag | Default | What it does |
| --- | --- | --- |
| `--tick-seconds` | `300` | Seconds between evaluation ticks |
| `--max-positions` | `5` | Cap on simultaneous open positions |
| `--max-daily-loss-pct` | `0.03` | Halt for the session if equity falls this much from start of day |
| `--pre-close-minutes` | `15` | No new entries within this many minutes of market close |

**Real-money flag** (requires both):

```bash
export ALPACA_LIVE=true
python agent.py --execute --live-money
```

Without `ALPACA_LIVE=true` in the environment, `--live-money` is
silently downgraded to paper trading with a warning. This is intentional.

**Run it as a daemon** (Linux/macOS):

```bash
nohup python agent.py --execute > agent.out 2>&1 &
echo $! > agent.pid                # save PID for later
```

Stop it:

```bash
touch AGENT_STOP                   # graceful — recommended
# or
kill -INT $(cat agent.pid)         # SIGINT, equivalent to Ctrl-C
```

### Native macOS desktop app (`desktop_app.py`) — dock-droppable

Same three sections as the web dashboard, but as a real native window
you can put on your Dock. Dark themed, modern flat UI, real-time
polling, no browser tab.

**Quick run (no install needed beyond venv):**

```bash
source venv/bin/activate
python desktop_app.py
```

**Dock launcher (drag-and-drop)** — simplest way to one-click launch:

```bash
chmod +x launcher.command
open .                # opens this folder in Finder
```

Then **drag `launcher.command` from the Finder window onto your Dock**.
Clicking it opens a small Terminal window and starts the app. Closing
the Terminal closes the app. (macOS treats `.command` files as
executable launchers by design.)

**Proper .app bundle** — for a real macOS application icon with no
Terminal window:

```bash
./build_macapp.sh
```

After ~2–5 min you'll get `dist/trading_enhancer.app`. Then:

```bash
open dist/trading_enhancer.app          # try it
mv dist/trading_enhancer.app /Applications/   # install
```

Now you can drag `trading_enhancer.app` from `/Applications` onto your
Dock just like any other app. The bundle is self-contained (~300–500 MB
because it ships its own Python + ML libraries — normal for
PyInstaller). The first launch from `/Applications` may prompt with
Gatekeeper; right-click → Open the first time to bypass.

**What you'll see** (all three sections in one window):

- **🩺 Health** — Broker / equity / positions / orders metric cards;
  a full Agent control panel (Start / Stop with options for dry run,
  tick seconds, data source); API key matrix; live log tail; recent
  signal records.
- **💹 Positions** — Color-coded P&L summary, table of open positions
  with avg entry / current / P&L $ / P&L %, table of recent orders.
- **🛒 Controls** — New bracket-order form (symbol / side radio /
  notional / stop% / take-profit% / confirmation), per-position Close
  buttons, per-order Cancel buttons, "Cancel ALL" button.

A background thread polls the broker every 8 seconds; the UI updates
without blocking. UTC clock in the status bar, transient toast
messages on every action.

### Web dashboard (`dashboard.py`)

A Streamlit single-file dashboard that gives you a live view of the
agent, broker, and signal pipeline — plus controls to place / close /
cancel trades by hand.

**Launch it** (requires the streamlit dep installed via `requirements.txt`):

```bash
streamlit run dashboard.py
```

The dashboard opens at http://localhost:8501.

Three tabs:

- **🩺 System Health** — broker connection status, agent process
  status, API key matrix, recent signals table, log tail.
- **💹 Active Trades** — open positions table with color-coded P&L
  bar chart, recent orders, account equity summary.
- **🛒 Trade Controls** — form to place a new bracket order
  (symbol / side / notional / stop% / take-profit%), per-position
  Close buttons, per-order Cancel buttons, "Cancel ALL" button.

Sidebar has an auto-refresh toggle (10-second poll) and a manual
refresh button. The dashboard reads from the **same** `AlpacaExecutor`
the agent + manual `trade.py` use, so every tab reflects identical
broker state.

**Safety note:** the new-order form requires you to tick an
"I have reviewed the parameters" checkbox before the submit button
becomes active. Real-money endpoint is only reachable if both
`--live-money` is wired in *and* `ALPACA_LIVE=true` is set in the
environment running the dashboard — same guard as `agent.py` and
`trade.py`.

## Extending the model

- **Drop in a real LSTM/XGBoost/Transformer.** `EnsemblePredictor` only
  requires `.predict(features) -> dict` with `direction`,
  `expected_return_pct`, `confidence`, and optionally `iv_change_pct`.
  Replace the heuristics in `run.py::_build_ensemble` once you have
  trained checkpoints under `models/`.
- **Add a vendor.** Implement a new fetcher in `data_loader.py` and
  surface it through `load_data()`. Keep the retry helper and the
  env‑var pattern.
- **Tighten risk.** Tune thresholds in `config/config.yaml` (the code
  reads them at startup) instead of editing constants in
  `risk_manager.py`.
- **New asset class.** Extend `data.universe` and, if needed, add a
  symbol→broker mapping in `run.py`.

See [`docs/architecture.md`](docs/architecture.md) for the full module
map and data contracts, and [`docs/deployment.md`](docs/deployment.md)
for VM provisioning, cron, and monitoring guidance.

### Switching brokers and promoting to a funded account

The execution layer is broker-agnostic. Pick the backend with
`--broker`:

```bash
python agent.py --execute --broker alpaca          # default (paper or live)
python agent.py --dry-run  --broker paper_only     # in-memory simulator
python agent.py --execute  --broker ibkr           # stub — implement first
```

**Live-money trading is gated by a track record.** Every session the
agent runs is appended to `data/track_record.jsonl`. Before
`AlpacaExecutor` will instantiate a live-money endpoint, the
[`PromotionGate`](src/execution/promotion_gate.py) reads the record
and verifies all of:

- ≥ 20 sessions logged
- All on paper (no prior live sessions)
- Net positive total P&L
- Per-session Sharpe-like score ≥ 0.5
- No single session lost > 5%
- No session triggered the daily loss breaker

If any criterion fails, instantiation raises `PromotionBlockedError`.
You cannot route live orders without proving paper profitability
first. To bypass (NOT recommended), set the environment variable
`TRADING_ENHANCER_BYPASS_GATE=I_ACCEPT_FULL_RESPONSIBILITY`. That
string is intentionally annoying so you'd have to actively look it up
in source.

### Portfolio-level risk countermeasures

Beyond the per-trade stops/take-profits in `src/risk/risk_manager.py`,
the agent now runs every potential entry through a stateful
[`CountermeasureSet`](src/risk/countermeasures.py) that enforces:

| Countermeasure | What it does |
| --- | --- |
| Circuit breaker | Halt new entries after N consecutive losses |
| Post-loss cooldown | Block entries for N minutes after any losing trade |
| Sector concentration cap | Max M positions per sector |
| Volatility regime | Shrink notional in high-VIX; refuse trades in extreme-VIX |
| Spread filter | Refuse to chase wide bid/ask spreads |
| Liquidity filter | Skip symbols below minimum 20-day volume |
| Session turnover cap | Max N trades per session |
| Slippage anomaly killer | Halt after 3 consecutive fills slipping > X bps |
| Daily blackout windows | Repeating UTC time-of-day blocks |
| Event blackouts | One-off UTC datetime windows (FOMC, earnings, etc.) |
| Trailing stop helper | `update_trailing_stop()` ratchets stops in your favor |
| Time-based stop helper | `time_stop_breached()` closes stale positions |

All are opt-in via [`CountermeasureConfig`](src/risk/countermeasures.py)
fields. The agent's defaults are conservative: 3-loss circuit breaker,
30-minute cooldown, 2 positions per sector, halt at VIX ≥ 30, 12 trades
per session.

### Training a real model (`train.py`)

The bundled ensemble in `run.py::_build_ensemble()` ships three
heuristic sub-models. The "xgboost" slot is the one designed to be
replaced by an actually-trained classifier. Run:

```bash
python train.py
```

This pulls 2 years of SPY history (via Alpaca if you have keys, else
Yahoo scraper), walks `FeatureEngineer` over it sample-by-sample,
labels each window with the forward 5-bar return (long if > +0.5%,
short if < -0.5%, drop otherwise — simplified triple-barrier method),
walk-forward CV-fits XGBoost, and saves three artifacts to `models/`:

| Artifact | What |
| --- | --- |
| `models/xgb.json` | Trained XGBoost model (JSON format) |
| `models/xgb_features.json` | Ordered feature column names for inference |
| `models/xgb_report.json` | CV scores, per-feature importance, label distribution |

The next `python run.py --mode live` or `python agent.py --execute`
will auto-detect these files and load the trained model in place of
the `_TabularSentiment` heuristic. Log line on startup:

    Ensemble using TRAINED XGBoost from models/xgb.json

To delete the trained model and revert to heuristic:

```bash
rm models/xgb.json models/xgb_features.json models/xgb_report.json
```

**Tuning flags:**

| Flag | Default | What |
| --- | --- | --- |
| `--symbols SPY QQQ AAPL` | `SPY` | Pool multiple tickers into the training set |
| `--lookback-days` | `730` | History per symbol |
| `--horizon` | `5` | Forward-return bars for labeling |
| `--long-threshold` | `0.005` | Forward return above this → long label |
| `--short-threshold` | `0.005` | Forward return below this (abs) → short label |
| `--n-splits` | `5` | Walk-forward CV folds |
| `--test-size` | `60` | Bars per CV test fold |
| `--n-estimators` | `400` | XGBoost trees |
| `--max-depth` | `4` | XGBoost depth |

**What "good" looks like:**

For short-horizon directional classification on liquid US equities
with cheaply-engineered features, mean CV accuracy of **0.52–0.55** is
the realistic ceiling. Anything dramatically above that is almost
certainly leakage (e.g., a feature that accidentally peeks at the
future) or overfitting — re-read papers 22, 55, 57 in
[`docs/trading_strategy_sources.md`](docs/trading_strategy_sources.md)
before celebrating. A 0.53 accuracy model with proper risk management
can be profitable. A claimed 0.75 model is almost certainly broken.

### Self-learning postmortem DB (`learn.py`)

A persistent, self-updating knowledge base of documented quant
failure modes. Seeded with 52 lessons from real history (LTCM,
Knight Capital, Quant Quake, Flash Crash, momentum crash, etc.)
plus published research on backtest overfitting, leakage,
survivorship bias, and behavioral mistakes. Auto-grows as the agent
runs.

**Three subsystems:**

1. **PostmortemDB** (`src/learning/postmortem_db.py`) — JSONL store
   at `data/postmortems.jsonl`. Each entry has id, category, title,
   description, symptom, mitigation, severity (1–5), source,
   confirmation count, references, tags.

2. **SessionObserver** (`src/learning/observer.py`) — runs at the
   end of every agent session. Confirms existing lessons that
   matched observed behavior (e.g., circuit breaker firing confirms
   the "revenge trading" lesson) and *writes new lessons* for novel
   patterns (e.g., "consecutive zero-trade sessions" or "session
   lost >2% without triggering breaker"). This is the
   self-updating half.

3. **Preflight** (`src/learning/preflight.py`) — runs before the
   agent starts a session or `train.py` starts training. Surfaces
   high-severity relevant lessons; blocks startup for known
   unmitigated conditions (e.g., requesting live money without a
   track record).

**CLI** — `learn.py`:

```bash
python learn.py init                              # bootstrap DB (idempotent)
python learn.py stats                             # counts by category/severity/source
python learn.py list --category overfitting       # browse by filter
python learn.py list --min-severity 5             # account-ending lessons
python learn.py show l_010                        # one lesson in full
python learn.py preflight agent                   # what would fire before live
python learn.py preflight agent --live-money      # plus live-money checks
python learn.py preflight train                   # before training a model
python learn.py add --id user_my_lesson           # add a custom lesson (interactive)
python learn.py export                            # dump DB → docs/postmortems.md
```

**Auto-update flow:**

```
agent session ends
       ↓
SessionObserver.observe(snapshot)
       ↓
matches against known lessons      →  db.confirm(l_xxx)
detects novel patterns             →  db.add_or_update(new_lesson)
firings logged to track_record     →  used by CorrelationAnalyzer
       ↓
data/postmortems.jsonl updated
       ↓
next preflight runs against the new state
```

After 10 agent runs you'll see `learn.py stats` show non-zero
`Self-confirmed` counts and possibly new `observer`-source lessons
in `learn.py list --source observer`. The system literally learns
from its own behavior every session.

**Outcome correlation analysis (`learn.py analyze`)** — turns "lesson
fired" into "does it actually correlate with losses?"

After you have a meaningful track record (10+ sessions with lesson
firings), run:

```bash
python learn.py analyze            # dry-run: report only
python learn.py analyze --apply    # commit suggested severity changes
```

For each lesson with ≥5 firings and ≥5 non-firings, it computes:

- mean session P&L when the lesson fired vs when it didn't
- effect size (Δ)
- one-tailed Welch t-test p-value
- categorisation:
  - **CONFIRMED NEGATIVE** — lesson genuinely correlates with worse
    outcomes in YOUR data → promote severity, harden the guardrail
  - **COUNTERMEASURE WORKING** — sessions where it fired actually
    outperformed (the guardrail caught real risk) → keep as-is
  - **NEUTRAL** — no measurable difference → consider loosening the
    associated countermeasure if it blocks aggressively

This closes the loop: the system isn't just adding lessons, it's
empirically validating which ones matter for *your specific*
strategy on *your specific* data, and adjusting severities
accordingly.

**Where preflight wires in:**

- `python agent.py …` → calls `run_preflight("agent", live_money=…)`
  before connecting to the broker. Prints a report; refuses to
  start if blockers fire.
- `python train.py …` → calls `run_preflight("train")` before
  loading data. Warns about PIT data, prior tuning runs, etc.

**Adding your own lessons** (e.g., after a personal lesson learned):

```bash
python learn.py add
# prompts for: id, category, title, description, symptom,
#               mitigation, severity, tags, references
```

The lesson is then consulted on every preflight and observation
just like the seed knowledge.

### Trading strategy research

Curated index of 120+ direct sources (academic papers, books, blogs,
open-source libraries, datasets, courses) for actually building a
trading strategy worth deploying:

[docs/trading_strategy_sources.md](docs/trading_strategy_sources.md)

Includes a "suggested reading order" — start with the **"Why most
systematic strategies fail"** section. The rest is only useful once
you've internalized the failure modes.

## Disclaimer

This repository is research scaffolding, not investment advice, and
**no part of it guarantees or implies profitability.** The bundled
ensemble sub-models are heuristic placeholders. Most quant strategies
underperform their benchmark over a decade. Paper-trade for 20+
sessions before the promotion gate will even permit live-money
trading — and even then, start with size you're prepared to lose.
