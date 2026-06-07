"""TradingEnv — Gymnasium environment for training an RL trading agent.

State
-----
A window of the last ``window`` bars of normalised technical features:
  ret_1, ret_5, ret_20, sma_spread, realized_vol, rsi, vol_z
plus two position scalars: [position (-1/0/1), unrealised_pnl_pct].

Total observation size: window × 7 + 2.

Actions
-------
0 = HOLD   — do nothing
1 = LONG   — open or stay long; close short if open
2 = SHORT  — open or stay short; close long if open

Reward
------
Step P&L in percent (long: Δclose/prev_close; short: -Δclose/prev_close)
minus a small transaction cost on every direction change.
Episode ends when all bars are consumed.

Usage
-----
    env = TradingEnv(df)                # df = OHLCV + feature columns
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


FEATURE_COLS = ["ret_1", "ret_5", "ret_20", "sma_spread", "realized_vol", "rsi", "vol_z"]
N_FEATURES   = len(FEATURE_COLS)   # 7 per bar
TRANSACTION_COST = 0.001           # 0.1% slippage per direction change


class TradingEnv(gym.Env):
    """Single-asset episodic trading environment.

    Parameters
    ----------
    df        : DataFrame with feature columns (from FeatureEngineer) and 'close'.
    window    : Number of bars to include in each observation.
    commission: Round-trip cost fraction applied on direction changes.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 20,
        commission: float = TRANSACTION_COST,
    ) -> None:
        super().__init__()
        self.df         = df.reset_index(drop=True)
        self.window     = window
        self.commission = commission
        self.n_steps    = len(df)

        # Fill missing feature columns with 0 so the env never crashes on sparse data.
        for col in FEATURE_COLS:
            if col not in self.df.columns:
                self.df[col] = 0.0
        self.df[FEATURE_COLS] = self.df[FEATURE_COLS].fillna(0.0)
        self.df["close"]      = self.df["close"].ffill().bfill()

        n_obs = window * N_FEATURES + 2   # +2: position scalar, unrealised P&L
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # 0=hold 1=long 2=short

        self._position: int  = 0      # -1 / 0 / 1
        self._entry_price: float = 0.0
        self._step: int  = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step        = self.window
        self._position    = 0
        self._entry_price = 0.0
        return self._obs(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_close = float(self.df.at[self._step - 1, "close"])
        curr_close = float(self.df.at[self._step,     "close"])

        # --- position management -----------------------------------------
        reward = 0.0
        new_position = self._action_to_position(action)

        if new_position != self._position:
            # Close existing position first
            if self._position != 0 and self._entry_price > 0:
                reward += self._pnl(self._position, self._entry_price, curr_close)
            # Deduct transaction cost on direction change
            if new_position != 0:
                reward -= self.commission
            self._position    = new_position
            self._entry_price = curr_close if new_position != 0 else 0.0
        else:
            # Running P&L for held position
            if self._position != 0 and self._entry_price > 0:
                reward += self._pnl(self._position, prev_close, curr_close)

        self._step += 1
        terminated = self._step >= self.n_steps - 1
        truncated  = False

        # Force-close at episode end
        if terminated and self._position != 0 and self._entry_price > 0:
            reward += self._pnl(
                self._position, self._entry_price, float(self.df.at[self._step - 1, "close"])
            )
            self._position = 0

        info = {
            "position":     self._position,
            "step":         self._step,
            "close":        curr_close,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        start = max(0, self._step - self.window)
        end   = self._step
        rows  = self.df[FEATURE_COLS].iloc[start:end].values

        # Pad with zeros if we're near the beginning
        if rows.shape[0] < self.window:
            pad  = np.zeros((self.window - rows.shape[0], N_FEATURES), dtype=np.float32)
            rows = np.vstack([pad, rows])

        flat = rows.astype(np.float32).flatten()

        # Unrealised P&L as a fraction
        if self._position != 0 and self._entry_price > 0:
            curr  = float(self.df.at[min(self._step, self.n_steps - 1), "close"])
            upnl  = self._pnl(self._position, self._entry_price, curr)
        else:
            upnl = 0.0

        extras = np.array([float(self._position), upnl], dtype=np.float32)
        obs = np.concatenate([flat, extras])
        return np.clip(obs, -10.0, 10.0)

    @staticmethod
    def _action_to_position(action: int) -> int:
        return {0: 0, 1: 1, 2: -1}[int(action)]

    @staticmethod
    def _pnl(position: int, entry: float, exit_: float) -> float:
        if entry == 0:
            return 0.0
        pct = (exit_ - entry) / entry
        return pct if position == 1 else -pct
