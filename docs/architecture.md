# Architecture

## High‑level data flow

```
┌──────────────┐   ┌─────────────────┐   ┌────────────────────┐
│  data_loader │──▶│  feature build  │──▶│   model.ensemble   │
│  (vendor IO) │   │  (windowing,    │   │   (LSTM / XGB /    │
│              │   │   indicators)   │   │    Transformer)    │
└──────────────┘   └─────────────────┘   └─────────┬──────────┘
                                                    │
                                                    ▼
                                          ┌────────────────────┐
                                          │  risk.risk_manager │
                                          │  (Kelly, stops,    │
                                          │   filters, DD)     │
                                          └─────────┬──────────┘
                                                    │
                                                    ▼
                                          ┌────────────────────┐
                                          │ signals.generator  │
                                          │ (JSON, ≥75% conf)  │
                                          └─────────┬──────────┘
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
                ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
                │  Live execution  │      │   Backtester     │      │   Persistence    │
                │  (broker API)    │      │   (walk‑forward) │      │   (CSV / DB)     │
                └──────────────────┘      └──────────────────┘      └──────────────────┘
```

## Module map

| Module | Responsibility |
| ------ | -------------- |
| `src/data_loader.py` | Vendor IO with retry/back‑off. OHLCV, options Greeks, news, macro. |
| `src/model/transformer_backbone.py` | 3‑layer Transformer, 256‑step window, direction + magnitude heads. |
| `src/model/sentiment_encoder.py` | FinBERT wrapper → score in [−1, 1] + salience flag. |
| `src/model/volatility_module.py` | GARCH(1,1) variance forecast + spline IV surface. |
| `src/model/ensemble.py` | Weighted aggregator (LSTM 0.30 / XGB 0.40 / Transformer 0.30). |
| `src/risk/risk_manager.py` | Kelly half‑sizing, ATR stops, TP, correlation / vol / blackout filters, drawdown monitor. |
| `src/signals/signal_generator.py` | Builds the JSON signal, enforces 75% confidence floor. |
| `src/backtest/backtest_runner.py` | Walk‑forward backtest, same risk manager as live. |
| `src/utils/logging_setup.py` | Rotating file + console logging. |
| `src/utils/cron_job.sh` | Daily 02:00 UTC scheduler entry. |
| `run.py` | CLI entry point (`--mode live | backtest`). |

## Design principles

- **Stateless risk manager.** Every function takes inputs and returns
  a decision; no hidden state. Makes live ↔ backtest parity trivial.
- **Sub‑model interchangeability.** `EnsemblePredictor` accepts any
  object exposing `.predict(features) -> dict`, so the LSTM / XGB /
  Transformer can be swapped without touching the aggregator.
- **Fail loud at boundaries.** Vendor failures retry with back‑off
  then raise; sub‑model contracts are enforced (`direction in {long,
  short}`); confidence is clamped to `[0, 1]`.
- **Config‑first.** All thresholds live in `config/config.yaml`; the
  code reads them rather than hard‑coding magic numbers.

## Data contracts

### Ensemble output
```json
{
  "direction":           "long",
  "expected_return_pct": 0.012,
  "iv_change_pct":       0.004,
  "confidence":          0.82,
  "components":          { "lstm": {...}, "xgboost": {...}, "transformer": {...} }
}
```

### Signal output
See [`signals/signal_generator.py`](../src/signals/signal_generator.py) for the
full schema. Required keys: `asset`, `timestamp`, `direction`,
`entry_price`, `stop_loss`, `take_profit`, `position_size_usd`,
`expected_return_pct`, `iv_change_pct`, `confidence`, `risk_flags`.
