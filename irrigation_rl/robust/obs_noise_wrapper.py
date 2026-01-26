from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import gymnasium as gym
import numpy as np


@dataclass
class ObsNoiseConfig:
    enabled: bool = True
    Dr_sigma_mm: float = 0.0        # 对 observation[0] 的噪声（Dr）
    theta_sigma: float = 0.0        # 对 observation[1] 的噪声（theta）
    ET0_sigma: float = 0.0          # 对 observation[2] 的噪声（ET0）
    clip_theta: Tuple[float, float] = (0.0, 1.0)


class ObsNoiseWrapper(gym.Wrapper):
    """
    只扰动 policy 输入的 observation，不修改 env 内部动力学。
    适用于：测试“传感器观测噪声 / 状态估计误差”的鲁棒性。

    假设 observation = [Dr, theta, ET0, stage_norm]
    """

    def __init__(self, env: gym.Env, cfg: ObsNoiseConfig, seed: int = 0):
        super().__init__(env)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs2, info2 = self._apply_noise(obs, info)
        return obs2, info2

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs2, info2 = self._apply_noise(obs, info)
        return obs2, reward, terminated, truncated, info2

    def _apply_noise(self, obs, info: Dict[str, Any]):
        if not self.cfg.enabled:
            return obs, info

        obs = np.array(obs, dtype=np.float32).copy()

        # obs[0]=Dr, obs[1]=theta, obs[2]=ET0
        if len(obs) >= 1 and self.cfg.Dr_sigma_mm > 0:
            obs[0] += self.rng.normal(0.0, self.cfg.Dr_sigma_mm)

        if len(obs) >= 2 and self.cfg.theta_sigma > 0:
            obs[1] += self.rng.normal(0.0, self.cfg.theta_sigma)
            obs[1] = np.clip(obs[1], self.cfg.clip_theta[0], self.cfg.clip_theta[1])

        if len(obs) >= 3 and self.cfg.ET0_sigma > 0:
            obs[2] += self.rng.normal(0.0, self.cfg.ET0_sigma)
            obs[2] = max(0.0, float(obs[2]))

        # 把“扰动后的观测”记录进 info，方便 debug/画图
        info["Dr_obs"] = float(obs[0]) if len(obs) >= 1 else None
        info["theta_obs"] = float(obs[1]) if len(obs) >= 2 else None
        info["ET0_obs"] = float(obs[2]) if len(obs) >= 3 else None

        return obs, info
