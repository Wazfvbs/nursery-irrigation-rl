from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
import gymnasium as gym


@dataclass
class ET0MultConfig:
    """
    Dynamics-side ET0 multiplicative perturbation.

    We apply:
      ET0_used(t) = ET0_base(t) * m_t

    where m_t is sampled from Uniform(low, high).

    per_day:
      - True: sample m_t i.i.d. each day (harder, "weather variability")
      - False: sample one constant multiplier per episode (domain shift)
    """
    enabled: bool = True
    per_day: bool = True
    low: float = 0.90
    high: float = 1.10


class ET0MultWrapper(gym.Wrapper):
    """
    Wrap an env that supports:
        env.unwrapped.set_domain_params(ET0_mult_series=np.ndarray)

    This wrapper generates ET0 multiplier sequence on reset() and injects it
    into the underlying env as `ET0_mult_series`.

    Reproducibility:
      - Uses its own RNG seeded by `seed` argument and reset(seed=...).
      - For a given seed, the generated sequence is deterministic.
    """

    def __init__(self, env: gym.Env, cfg: ET0MultConfig, seed: int = 0):
        super().__init__(env)
        self.cfg = cfg
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        # Try to determine horizon length (fallback to 365)
        self.horizon_days = self._infer_horizon_days()

    def _infer_horizon_days(self) -> int:
        # Attempt to read from unwrapped env config
        try:
            if hasattr(self.env.unwrapped, "cfg") and hasattr(self.env.unwrapped.cfg, "horizon_days"):
                return int(self.env.unwrapped.cfg.horizon_days)
        except Exception:
            pass
        return 365

    def _inject_series(self, series: np.ndarray) -> None:
        """
        Inject ET0 multiplier series into base env.
        Base env must implement set_domain_params(ET0_mult_series=...).
        """
        if not hasattr(self.env.unwrapped, "set_domain_params"):
            raise RuntimeError(
                "Base env does not implement set_domain_params(...). "
                "Please update irrigation_rl/envs/nursery_env.py with set_domain_params support."
            )
        try:
            self.env.unwrapped.set_domain_params(ET0_mult_series=series)
        except TypeError as e:
            raise RuntimeError(
                "set_domain_params exists but does not accept ET0_mult_series. "
                "Please update nursery_env.set_domain_params signature."
            ) from e

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Update RNG for reproducibility (match gym reset convention)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        # If disabled, clear series injection (use default ET0_mult=1)
        if not self.cfg.enabled:
            try:
                self.env.unwrapped.set_domain_params(ET0_mult_series=None, ET0_mult=1.0)
            except Exception:
                pass
            return self.env.reset(seed=seed, options=options)

        low = float(self.cfg.low)
        high = float(self.cfg.high)
        if high < low:
            low, high = high, low

        if self.cfg.per_day:
            series = self.rng.uniform(low, high, size=(self.horizon_days,)).astype(np.float32)
        else:
            m = float(self.rng.uniform(low, high))
            series = (np.ones((self.horizon_days,), dtype=np.float32) * m)

        self._inject_series(series)

        obs, info = self.env.reset(seed=seed, options=options)

        # Helpful debug signals in info
        if isinstance(info, dict):
            info = dict(info)
            info["robust_et0_enabled"] = int(True)
            info["robust_et0_low"] = float(low)
            info["robust_et0_high"] = float(high)
            info["robust_et0_per_day"] = int(bool(self.cfg.per_day))
        return obs, info

    def step(self, action):
        return self.env.step(action)
