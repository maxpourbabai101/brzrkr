"""Train Transformer model on FeatureEngineer output."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_engineer import FeatureEngineer
from src.model.transformer_backbone import TransformerPredictor, TransformerConfig
from src.utils.logging_setup import configure_logging

logger = logging.getLogger("trading_enhancer.train_transformer")


def _fetch_prices(symbol: str, lookback_days: int, *, prefer_scraper: bool = False) -> pd.DataFrame:
    if not prefer_scraper:
        try:
            from src.data_loader import fetch_futures
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=lookback_days)
            df = fetch_futures(symbol, start, end)
            if not df.empty and len(df) > 50:
                return df
        except Exception as exc:
            logger.info("Alpaca fetch failed for %s (%s); trying scraper.", symbol, exc)

    from src.data_scraper import WebDataScraper
    agent = WebDataScraper()
    df = agent.scrape_ohlcv(symbol, range_=_lookback_to_yahoo_range(lookback_days))
    if df.empty:
        raise RuntimeError(f"Could not fetch any price history for {symbol}")
    return df


def _lookback_to_yahoo_range(days: int) -> str:
    if days <= 30: return "1mo"
    if days <= 90: return "3mo"
    if days <= 180: return "6mo"
    if days <= 365: return "1y"
    if days <= 730: return "2y"
    if days <= 1825: return "5y"
    return "max"  # use max instead of 10y for more data


def _create_sequences(features: pd.DataFrame, horizon: int, long_thresh: float, short_thresh: float):
    prices = features[['close']].copy()
    N = len(features)
    seq_len = features.attrs.get('seq_len', 256)

    feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != 'close']

    X, y = [], []
    for t in range(seq_len, N - horizon):
        window_df = features[feature_cols].iloc[t - seq_len:t]
        window = window_df.values
        entry = float(prices['close'].iloc[t])
        exit_ = float(prices['close'].iloc[t + horizon])
        fwd_ret = (exit_ / entry) - 1.0

        if fwd_ret > long_thresh:
            label = 1
        elif fwd_ret < -short_thresh:
            label = 0
        else:
            continue

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y), feature_cols


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Transformer on FeatureEngineer output")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "AAPL", "NVDA"])
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--long-threshold", type=float, default=0.005)
    parser.add_argument("--short-threshold", type=float, default=0.005)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--prefer-scraper", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    configure_logging(level=args.log_level, log_path="trading_enhancer.log")

    fe = FeatureEngineer(window=args.seq_len)
    all_X, all_y = [], []
    all_feature_cols = None

    for i, symbol in enumerate(args.symbols, 1):
        print(f"  [{i}/{len(args.symbols)}] {symbol}: fetching...", end="", flush=True)
        prices = _fetch_prices(symbol, args.lookback_days, prefer_scraper=args.prefer_scraper)
        print(f" got {len(prices)} bars; building features...", end="", flush=True)
        bundle = {"prices": prices}
        features = fe.build_features(bundle, full_history=True)
        if features.empty or len(features) < args.seq_len + args.horizon + 1:
            print(" insufficient data")
            continue

        X, y, feature_cols = _create_sequences(features, args.horizon, args.long_threshold, args.short_threshold)
        print(f" {len(X)} samples")
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        if all_feature_cols is None:
            all_feature_cols = feature_cols

    if not all_X:
        logger.error("No training data produced.")
        return 2

    feature_cols = all_feature_cols

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    print(f"Combined: {X.shape[0]} samples, {X.shape[2]} features, long ratio: {y.mean():.2%}")

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)

    config = TransformerConfig(
        input_dim=X.shape[2],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        seq_len=args.seq_len,
    )
    model = TransformerPredictor(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits, _ = model(xb)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "transformer_model.pt")
            config_dict = {
                "input_dim": config.input_dim,
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "dim_feedforward": config.dim_feedforward,
                "dropout": config.dropout,
                "seq_len": config.seq_len,
            }
            (output_dir / "transformer_config.json").write_text(json.dumps(config_dict, indent=2))
            (output_dir / "transformer_features.json").write_text(json.dumps(feature_cols, indent=2))
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Artifacts written to: {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())