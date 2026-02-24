from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from .randomization import RandomizationConfig, apply_domain_randomization


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Episode-level domain randomization wrapper.

    On each reset, sample ET0/Kc/Zr multipliers and inject into base env via:
      env.unwrapped.set_domain_params(...)
    """

    def __init__(self, env: gym.Env, cfg: RandomizationConfig, seed: int = 0):
        super().__init__(env)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        params = apply_domain_randomization({}, self.cfg, self.rng)
        if hasattr(self.env.unwrapped, "set_domain_params"):
            self.env.unwrapped.set_domain_params(
                ET0_mult=float(params.get("ET0_mult", 1.0)),
                Kc_mult=float(params.get("Kc_mult", 1.0)),
                Zr_mult=float(params.get("Zr_mult", 1.0)),
                ET0_mult_series=None,
            )

        obs, info = self.env.reset(seed=seed, options=options)
        if isinstance(info, dict):
            info = dict(info)
            info["dr_enabled"] = int(bool(self.cfg.enabled))
            info["dr_ET0_mult"] = float(params.get("ET0_mult", 1.0))
            info["dr_Kc_mult"] = float(params.get("Kc_mult", 1.0))
            info["dr_Zr_mult"] = float(params.get("Zr_mult", 1.0))
        return obs, info

    def step(self, action):
        return self.env.step(action)
