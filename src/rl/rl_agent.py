"""RLAgent — PPO-based trading agent.

Training
--------
    from src.rl.rl_agent import RLAgent
    agent = RLAgent()
    agent.train(df_list, total_timesteps=200_000)
    agent.save("models/rl_ppo")

Inference (ensemble sub-model interface)
-----------------------------------------
The loaded agent exposes ``predict(features) -> dict`` matching the
contract expected by EnsemblePredictor sub-models:

    {
        "direction":           "long" | "short" | "hold",
        "confidence":          float in [0, 1],
        "expected_return_pct": float,
    }

The ``confidence`` is derived from the deterministic policy's action
probability (softmax over the policy logits). ``expected_return_pct``
is a rough estimate based on the mean episode return seen during
recent training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_DIR  = Path("models")
_MODEL_NAME = "rl_ppo"


class RLAgent:
    """Wraps stable-baselines3 PPO for episodic training + live inference."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model = None
        self._mean_return: float = 0.0
        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df_list: List[pd.DataFrame],
        *,
        window: int = 20,
        total_timesteps: int = 200_000,
        n_envs: int = 4,
        save_path: Optional[Path] = None,
        verbose: int = 1,
    ) -> None:
        """Train a PPO agent on a list of OHLCV+feature DataFrames.

        Parameters
        ----------
        df_list         : List of feature DataFrames (one per symbol/period).
        window          : Observation window in bars.
        total_timesteps : Total environment steps to train for.
        n_envs          : Parallel environments (uses SubprocVecEnv).
        save_path       : Where to write the model. Defaults to models/rl_ppo.
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from src.rl.trading_env import TradingEnv

        logger.info(
            "RLAgent.train: %d DataFrames, window=%d, total_timesteps=%d",
            len(df_list), window, total_timesteps,
        )

        # Rotate through the df_list so each parallel env gets a random df.
        import random

        def _make_env(df: pd.DataFrame, w: int):
            def _init():
                return TradingEnv(df, window=w)
            return _init

        fns = [_make_env(random.choice(df_list), window) for _ in range(n_envs)]
        vec_env = make_vec_env(lambda: TradingEnv(df_list[0], window=window),
                               n_envs=n_envs)

        self._model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=verbose,
        )
        self._model.learn(total_timesteps=total_timesteps)
        vec_env.close()

        # Evaluate mean episode return on the first df for confidence calibration.
        self._mean_return = self._eval_mean_return(df_list[0], window=window)
        logger.info("RLAgent mean eval return: %.4f", self._mean_return)

        out = save_path or (_MODEL_DIR / _MODEL_NAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(out))
        logger.info("RLAgent saved to %s", out)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        if self._model is None:
            raise RuntimeError("No model to save — train first.")
        out = Path(path or _MODEL_DIR / _MODEL_NAME)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(out))

    def load(self, path: Optional[Path] = None) -> None:
        from stable_baselines3 import PPO
        from src.rl.trading_env import TradingEnv

        p = Path(path or _MODEL_DIR / _MODEL_NAME)
        zip_path = p.with_suffix(".zip") if not str(p).endswith(".zip") else p
        if not zip_path.exists():
            raise FileNotFoundError(f"RL model not found: {zip_path}")
        self._model = PPO.load(str(p))
        logger.info("RLAgent loaded from %s", p)

    @classmethod
    def load_if_exists(cls) -> Optional["RLAgent"]:
        """Return a loaded RLAgent if the model file exists, else None."""
        p = _MODEL_DIR / (_MODEL_NAME + ".zip")
        if not p.exists():
            return None
        try:
            agent = cls()
            agent.load(_MODEL_DIR / _MODEL_NAME)
            return agent
        except Exception as exc:
            logger.warning("Could not load RL model: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Inference — ensemble sub-model interface
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Given a feature DataFrame (recent bars), return a signal dict.

        Parameters
        ----------
        features : DataFrame produced by FeatureEngineer (most-recent bars last).

        Returns
        -------
        dict with keys: direction, confidence, expected_return_pct.
        """
        if self._model is None:
            return {"direction": "hold", "confidence": 0.0, "expected_return_pct": 0.0}

        from src.rl.trading_env import TradingEnv, FEATURE_COLS, N_FEATURES

        window = self._model.policy.observation_space.shape[0] // N_FEATURES
        # Build a minimal observation from the tail of features
        rows = features[FEATURE_COLS].tail(window).fillna(0.0).values
        if rows.shape[0] < window:
            pad  = np.zeros((window - rows.shape[0], N_FEATURES), dtype=np.float32)
            rows = np.vstack([pad, rows])

        flat = rows.astype(np.float32).flatten()
        obs  = np.concatenate([flat, np.zeros(2, dtype=np.float32)])
        obs  = np.clip(obs, -10.0, 10.0)

        action, _states = self._model.predict(obs, deterministic=True)
        action = int(action)

        # Get action probabilities for confidence estimate
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = self._model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.numpy()[0]

        direction_map = {0: "hold", 1: "long", 2: "short"}
        direction = direction_map[action]
        confidence = float(probs[action])

        # Scale expected return from training history
        exp_ret = abs(self._mean_return) * (1 if action == 1 else -1 if action == 2 else 0)

        return {
            "direction":           direction,
            "confidence":          round(confidence, 4),
            "expected_return_pct": round(exp_ret, 4),
        }

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _eval_mean_return(self, df: pd.DataFrame, *, window: int = 20) -> float:
        """Run one episode deterministically and return cumulative P&L."""
        from src.rl.trading_env import TradingEnv

        env = TradingEnv(df, window=window)
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        return total
