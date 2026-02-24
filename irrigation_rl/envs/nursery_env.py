from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .fao56 import (
    calc_TAW,
    calc_RAW,
    calc_Ks,
    Dr_to_theta,
    calc_ET0_PM,
    calc_ET0_fallback,
)
from .dynamics import WaterBalanceInputs, update_Dr
from .weather import WeatherProvider


# ============================================================
# Config
# ============================================================
@dataclass
class EnvConfig:
    """Nursery FAO-56 Root-zone env config"""

    horizon_days: int = 90
    a_max_mm: float = 15.0
    dt_days: float = 1.0

    theta_fc: float = 0.30
    theta_wp: float = 0.12
    Zr_m: float = 0.30
    p: float = 0.50

    # Kc schedule
    Kc_ini: float = 0.50
    Kc_mid: float = 1.00
    Kc_end: float = 0.80
    stage_ini_days: int = 20
    stage_mid_days: int = 50
    stage_end_days: int = 20

    terminate_on_theta_below_wp: bool = True
    terminate_on_Dr_above_TAW: bool = True


def kc_by_stage(cfg: EnvConfig, day: int) -> float:
    """Piecewise Kc schedule"""
    if day < cfg.stage_ini_days:
        return cfg.Kc_ini
    if day < cfg.stage_ini_days + cfg.stage_mid_days:
        return cfg.Kc_mid
    return cfg.Kc_end


def stage_norm(cfg: EnvConfig, day: int) -> float:
    """Normalize stage progress to [0,1]"""
    return float(np.clip(day / max(cfg.horizon_days - 1, 1), 0.0, 1.0))


# ============================================================
# Env
# ============================================================
class NurseryIrrigationEnv(gym.Env):
    """
    FAO-56 Plant Nursery Irrigation Environment (Dr-based)

    State (observation):
      [Dr(mm), theta(m3/m3), ET0(mm/day), stage_norm]

    Action:
      irrigation I_t in [0, a_max] (mm/day)

    Notes:
    - Reward is injected externally by RewardWrapper (training-time).
    - We expose rich info for logging & plotting & paper tables.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, weather: WeatherProvider, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.weather = weather
        self.rng = np.random.default_rng(seed)

        # FAO-56 constants
        self.TAW = float(calc_TAW(cfg.theta_fc, cfg.theta_wp, cfg.Zr_m))
        self.RAW = float(calc_RAW(cfg.p, self.TAW))

        # Observation: Dr, theta, ET0, stage_norm
        obs_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([max(self.TAW, 1.0), 1.0, 30.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: [0, a_max]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([cfg.a_max_mm], dtype=np.float32),
            dtype=np.float32,
        )

        # Runtime states
        self.day: int = 0
        self.Dr: float = 0.0
        self.prev_Dr: float = 0.0
        self.last_I: float = 0.0

        # cached daily variables (debug/paper)
        self.last_ET0: float = 0.0
        self.last_ETc: float = 0.0
        self.last_Kc: float = 0.0
        self.last_Ks: float = 0.0
        self.last_DP: float = 0.0

    # -----------------------
    # Core helpers
    # -----------------------
    def _safe_ET0(self, w: Dict[str, Any]) -> float:
        """
        ET0 calculator:
        - Prefer PM (Penman–Monteith) if inputs exist
        - Fallback if missing keys / errors

        calc_ET0_PM 内部也可以做 fallback，但这里加一层防御更稳健。
        """
        try:
            et0 = float(calc_ET0_PM(w))
            if not np.isfinite(et0) or et0 < 0.0:
                raise ValueError("ET0 invalid")
            return et0
        except Exception:
            et0 = float(calc_ET0_fallback(w))
            if not np.isfinite(et0) or et0 < 0.0:
                et0 = 0.0
            return et0

    def _get_obs(self) -> np.ndarray:
        # Next-state observation uses current self.day (already updated)
        w = self.weather.get_day(self.day)
        ET0 = self._safe_ET0(w)
        theta = float(Dr_to_theta(self.Dr, self.cfg.theta_fc, self.cfg.Zr_m))
        s = stage_norm(self.cfg, self.day)
        return np.array([self.Dr, theta, ET0, s], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        theta = float(Dr_to_theta(self.Dr, self.cfg.theta_fc, self.cfg.Zr_m))
        s = stage_norm(self.cfg, self.day)

        return {
            "day": int(self.day),
            "stage_norm": float(s),  # ✅ 关键：RewardWrapper 的 dynamic target 用它
            "Dr_mm": float(self.Dr),
            "theta": float(theta),
            "TAW_mm": float(self.TAW),
            "RAW_mm": float(self.RAW),
            # cached daily vars (after step they are updated)
            "ET0": float(self.last_ET0),
            "ETc": float(self.last_ETc),
            "Kc": float(self.last_Kc),
            "Ks": float(self.last_Ks),
            "DP": float(self.last_DP),
            "I_mm": float(self.last_I),
        }

    # -----------------------
    # Gym API
    # -----------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # weather provider reset
        self.weather.reset(seed=seed if seed is not None else 0)

        self.day = 0

        # Initialize near moderate depletion (random in [0.2TAW, 0.6TAW])
        self.Dr = float(self.rng.uniform(0.2 * self.TAW, 0.6 * self.TAW))
        self.prev_Dr = self.Dr
        self.last_I = 0.0

        # clear caches
        self.last_ET0 = 0.0
        self.last_ETc = 0.0
        self.last_Kc = 0.0
        self.last_Ks = 0.0
        self.last_DP = 0.0

        obs = self._get_obs()
        info = self._get_info()  # includes stage_norm=0.0
        return obs, info

    def step(self, action):
        """
        One-day transition:
        - Use current day weather and crop stage
        - Apply irrigation action
        - Update Dr using FAO-56-like water balance
        - Advance day
        """
        # --- robust action parsing ---
        if isinstance(action, (float, int)):
            a = float(action)
        else:
            a = float(np.asarray(action).reshape(-1)[0])

        I = float(np.clip(a, 0.0, self.cfg.a_max_mm))

        # current day weather
        w = self.weather.get_day(self.day)

        # ET0 / crop / stress
        ET0 = self._safe_ET0(w)
        Kc = float(kc_by_stage(self.cfg, self.day))
        Ks = float(calc_Ks(self.Dr, self.RAW, self.TAW))
        ETc = float(Kc * Ks * ET0)

        # ------------------------------------------------------------
        # Minimal DP model (paper-friendly placeholder)
        # Intuition:
        #   If irrigation is so large that Dr would become negative,
        #   the surplus percolates as deep drainage DP and Dr is clamped at 0.
        #
        # This gives DP a physically consistent meaning and non-zero possibility.
        # ------------------------------------------------------------
        # Without rainfall etc., Dr_raw ≈ Dr + ETc - I
        Dr_raw_noDP = float(self.Dr + ETc - I)
        DP = float(max(0.0, -Dr_raw_noDP))  # surplus water percolates
        wb = WaterBalanceInputs(P=0.0, RO=0.0, CR=0.0, DP=DP)

        Dr_next = float(update_Dr(self.Dr, ETc, I, wb, self.TAW))

        # update internal states
        self.prev_Dr = self.Dr
        self.Dr = Dr_next
        self.last_I = I

        # cache daily vars for info
        self.last_ET0 = ET0
        self.last_ETc = ETc
        self.last_Kc = Kc
        self.last_Ks = Ks
        self.last_DP = DP

        # advance day (next observation will use day+1)
        self.day += 1

        # termination / truncation
        terminated = False
        theta_next = float(Dr_to_theta(self.Dr, self.cfg.theta_fc, self.cfg.Zr_m))

        if self.cfg.terminate_on_theta_below_wp and theta_next < self.cfg.theta_wp:
            terminated = True
        if self.cfg.terminate_on_Dr_above_TAW and self.Dr >= self.TAW:
            terminated = True

        truncated = bool(self.day >= self.cfg.horizon_days)

        # reward computed externally by RewardWrapper
        reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info
