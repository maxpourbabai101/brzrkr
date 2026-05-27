# Deployment guide

## 1. Provision

- A Linux VM with Python ≥ 3.10, ≥ 4 GB RAM (8 GB recommended for the
  FinBERT model), and outbound HTTPS to vendor APIs.
- A non‑root user, e.g. `trader`, that owns `/opt/trading_enhancer`.

```bash
sudo adduser --system --group trader
sudo mkdir -p /opt/trading_enhancer
sudo chown trader:trader /opt/trading_enhancer
sudo -u trader git clone <your fork> /opt/trading_enhancer
```

## 2. Install

```bash
sudo -u trader bash -c "cd /opt/trading_enhancer && ./setup.sh"
```

`setup.sh` creates the virtualenv, installs pinned dependencies from
`requirements.txt`, and runs an import smoke test.

## 3. Configure secrets

Either:

**Option A — `.env` file (preferred for cron):**
```bash
sudo -u trader tee /opt/trading_enhancer/.env <<'EOF'
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
TRADIER_API_KEY=...
NEWSAPI_KEY=...
FRED_API_KEY=...
EOF
sudo chmod 600 /opt/trading_enhancer/.env
```

**Option B — systemd EnvironmentFile** (for `systemd` services).

## 4. Schedule the daily run

Edit `src/utils/cron_job.sh`, set `PROJECT_DIR=/opt/trading_enhancer`,
then install the entry:

```bash
chmod +x /opt/trading_enhancer/src/utils/cron_job.sh
( crontab -u trader -l 2>/dev/null;
  echo "0 2 * * * /opt/trading_enhancer/src/utils/cron_job.sh"
) | sudo crontab -u trader -
```

This runs at 02:00 UTC every day. Adjust as needed for your asset
class / exchange hours.

## 5. Monitor

- `trading_enhancer.log` (rotated, 10 MB × 5) — application log.
- `cron.log` — captured stdout/stderr from the cron entry.
- `data/signals/` — emitted JSON signals, one file per run.
- `data/backtest_out/equity_curve.csv`, `trades.csv` — backtest artifacts.

Recommended alerts:
- `grep -E 'ERROR|Drawdown breach' trading_enhancer.log | tail`
- Equity floor breach (compare current equity to
  `live.account_equity_floor_usd` in `config/config.yaml`).

## 6. Upgrades

```bash
cd /opt/trading_enhancer
sudo -u trader git pull
sudo -u trader bash -c "source venv/bin/activate && pip install -r requirements.txt"
sudo -u trader bash -c "source venv/bin/activate && pytest -q"
```

If the test suite fails, **do not** roll the cron forward — keep the
previous virtualenv until the breakage is understood.

## 7. Disaster recovery

- All vendor calls are idempotent reads; rerunning is safe.
- Trades are placed via the broker API and persisted server‑side.
- Local state worth backing up: `models/` (checkpoints), `data/signals/`,
  `data/backtest_out/`, and any custom config under `config/`.
