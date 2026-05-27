"""Vector Store — compressed market snapshot storage & fast retrieval.

Saves and loads 32-dim feature vectors for every scanned symbol in a
compact numpy .npz file.  Also maintains a human-readable latest.json
summary for quick UI rendering.

Storage layout
──────────────
  data/market_vectors/
      latest.json          — full snapshot list (JSON, fast UI read)
      YYYYMMDD_HHMMSS.npz  — compressed numpy vectors + metadata
      (files older than MAX_AGE_DAYS are auto-deleted on save)

Key methods
───────────
  VectorStore().save_snapshot(snapshots)          → write to disk
  VectorStore().load_latest()                     → list[dict]
  VectorStore().get_top_setups(category, n)       → filtered ranked list
  VectorStore().find_similar(vector, k)           → cosine-similar symbols
  VectorStore().cache_age_minutes()               → minutes since last save
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent.parent
VECTOR_DIR = ROOT / "data" / "market_vectors"
LATEST     = VECTOR_DIR / "latest.json"
MAX_AGE_DAYS = 30
FEATURE_DIM  = 32


class VectorStore:
    """Read/write compressed market vectors + fast ranked lookups."""

    def __init__(self) -> None:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────

    def save_snapshot(self, snapshots) -> None:
        """Persist a list of SymbolSnapshot objects.

        Writes both a compressed .npz (for ML/similarity) and a flat
        latest.json (for instant UI reads).
        """
        if not snapshots:
            return

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # ── Build arrays ──────────────────────────────────────────────
        symbols  = [s.symbol for s in snapshots]
        metadata = [s.to_dict() for s in snapshots]
        vectors  = np.zeros((len(snapshots), FEATURE_DIM), dtype=np.float32)
        for i, s in enumerate(snapshots):
            if s.features and len(s.features) == FEATURE_DIM:
                vectors[i] = s.features

        # ── Save compressed npz ───────────────────────────────────────
        npz_path = VECTOR_DIR / f"{ts_str}.npz"
        np.savez_compressed(
            npz_path,
            symbols  = np.array(symbols, dtype="U10"),
            vectors  = vectors,
        )

        # ── Save latest.json (snapshot list + save time) ──────────────
        payload = {
            "saved_at":   datetime.now(timezone.utc).isoformat(),
            "count":      len(snapshots),
            "symbols":    symbols,
            "snapshots":  metadata,
        }
        LATEST.write_text(json.dumps(payload, indent=2))

        # ── Cleanup old files ─────────────────────────────────────────
        self._cleanup()
        logger.info("VectorStore: saved %d vectors → %s", len(snapshots), npz_path.name)

    # ── Read ──────────────────────────────────────────────────────────

    def load_latest(self) -> List[Dict[str, Any]]:
        """Load latest.json. Returns [] if no data yet."""
        if not LATEST.exists():
            return []
        try:
            data = json.loads(LATEST.read_text())
            return data.get("snapshots", [])
        except Exception as exc:
            logger.warning("VectorStore.load_latest failed: %s", exc)
            return []

    def load_latest_meta(self) -> Optional[Dict]:
        """Return the full metadata dict (saved_at, count, etc.)."""
        if not LATEST.exists():
            return None
        try:
            return json.loads(LATEST.read_text())
        except Exception:
            return None

    def cache_age_minutes(self) -> float:
        """Minutes since last save.  Returns 9999 if no cache."""
        if not LATEST.exists():
            return 9999.0
        try:
            meta = json.loads(LATEST.read_text())
            saved = datetime.fromisoformat(meta["saved_at"])
            now   = datetime.now(timezone.utc)
            # handle naive datetimes
            if saved.tzinfo is None:
                from datetime import timezone as tz
                saved = saved.replace(tzinfo=tz.utc)
            delta = (now - saved).total_seconds() / 60
            return max(0.0, delta)
        except Exception:
            return 9999.0

    # ── Ranked lookups ────────────────────────────────────────────────

    def get_top_setups(self, category: str = "ALL",
                       n: int = 10,
                       signal: Optional[str] = None) -> List[Dict]:
        """Return top-N symbols by |score| for a given category.

        category: asset_type string or "ALL" or special keys:
                  "FUTURES", "OPTIONS", "SWING", "SCALP"
        signal:   None | "bullish" | "bearish"
        """
        snaps = self.load_latest()
        if not snaps:
            return []

        _FUTURES_TYPES  = {"COMMODITY_ETF", "BOND_ETF", "LEVERAGED_ETF",
                           "VOLATILITY", "INDEX_ETF"}
        _OPTIONS_TYPES  = {"STOCK", "INDEX_ETF", "CRYPTO_ETF"}
        _SWING_TYPES    = {"STOCK", "INDEX_ETF", "SECTOR_ETF"}
        _SCALP_TYPES    = {"LEVERAGED_ETF", "VOLATILITY", "CRYPTO_ETF"}

        category_map = {
            "FUTURES": _FUTURES_TYPES,
            "OPTIONS": _OPTIONS_TYPES,
            "SWING":   _SWING_TYPES,
            "SCALP":   _SCALP_TYPES,
        }

        if category == "ALL":
            filtered = snaps
        elif category in category_map:
            allowed = category_map[category]
            filtered = [s for s in snaps if s.get("asset_type") in allowed]
        else:
            filtered = [s for s in snaps if s.get("asset_type") == category]

        if signal:
            filtered = [s for s in filtered
                        if s.get("signal", "").lower() == signal.lower()]

        # For OPTIONS: prioritize extreme RSI + high vol ratio
        if category == "OPTIONS":
            def _opts_score(s):
                rsi = s.get("rsi", 50)
                vr  = s.get("vol_ratio", 1)
                return abs(s.get("score", 0)) + abs(rsi - 50) * 0.5 + (vr - 1) * 10
            filtered.sort(key=_opts_score, reverse=True)
        else:
            filtered.sort(key=lambda s: abs(s.get("score", 0)), reverse=True)

        return filtered[:n]

    def get_all_scored(self) -> List[Dict]:
        """All snapshots, sorted by |score| descending."""
        snaps = self.load_latest()
        snaps.sort(key=lambda s: abs(s.get("score", 0)), reverse=True)
        return snaps

    # ── Vector similarity ─────────────────────────────────────────────

    def find_similar(self, query_vector: np.ndarray,
                     top_k: int = 10,
                     exclude_symbol: str = "") -> List[Dict]:
        """Find top-k symbols with highest cosine similarity to query.

        Useful for 'what other setups look like this one right now'.
        """
        # Load latest .npz
        npz_files = sorted(VECTOR_DIR.glob("*.npz"), reverse=True)
        if not npz_files:
            return []
        try:
            data = np.load(npz_files[0])
            symbols = data["symbols"]
            vectors = data["vectors"]
        except Exception:
            return []

        # Cosine similarity
        qv = query_vector.astype(np.float32)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        sims = (vectors @ qv) / (
            np.linalg.norm(vectors, axis=1) * qn + 1e-9
        )

        # Sort descending
        idx = np.argsort(sims)[::-1]
        snaps = {s["symbol"]: s for s in self.load_latest()}
        results = []
        for i in idx:
            sym = str(symbols[i])
            if sym == exclude_symbol:
                continue
            s = snaps.get(sym)
            if s:
                s = dict(s)
                s["similarity"] = round(float(sims[i]), 3)
                results.append(s)
            if len(results) >= top_k:
                break
        return results

    # ── Cleanup ───────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Delete .npz files older than MAX_AGE_DAYS."""
        now = datetime.now(timezone.utc)
        for f in VECTOR_DIR.glob("*.npz"):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
                if (now - mtime).days > MAX_AGE_DAYS:
                    f.unlink()
                    logger.debug("VectorStore: deleted old file %s", f.name)
            except Exception:
                pass

    # ── Diagnostics ───────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary string."""
        meta = self.load_latest_meta()
        if meta is None:
            return "No market data collected yet."
        age  = self.cache_age_minutes()
        cnt  = meta.get("count", 0)
        saved = meta.get("saved_at", "?")[:16]
        snaps = meta.get("snapshots", [])
        bulls  = sum(1 for s in snaps if s.get("signal") == "bullish")
        bears  = sum(1 for s in snaps if s.get("signal") == "bearish")
        alerts = sum(1 for s in snaps if abs(s.get("score", 0)) > 50)
        return (
            f"  {cnt} symbols  ·  {age:.0f} min ago  ·  saved {saved}\n"
            f"  {bulls} bullish  ·  {bears} bearish  ·  {alerts} high-conviction"
        )
