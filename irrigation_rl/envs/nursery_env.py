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

    Observation:
      [Dr(mm), theta(m3/m3), ET0(mm/day), stage_norm]

    Action:
      irrigation I_t in [0, a_max] (mm/day)

    Notes:
    - Reward is injected externally by RewardWrapper (training-time).
    - We expose rich info for logging & plotting & paper tables.
    - Domain perturbations (ET0/Kc/Zr multipliers) can be injected at episode-level
      via `set_domain_params(...)`. This is used by robustness evaluation and
      (optionally) domain randomization training.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, weather: WeatherProvider, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.weather = weather
        self.rng = np.random.default_rng(seed)

        # -----------------------
        # Domain parameters (episode-level)
        # -----------------------
        # Multipliers apply as:
        #   ET0_used(t) = ET0_base(t) * ET0_mult_today(t)
        #   Kc_used(t)  = Kc_stage(t) * Kc_mult
        #   Zr_used     = cfg.Zr_m * Zr_mult  (affects TAW/RAW & theta mapping)
        self.domain_params: Dict[str, float] = {"ET0_mult": 1.0, "Kc_mult": 1.0, "Zr_mult": 1.0}
        self._ET0_mult_series: Optional[np.ndarray] = None  # per-day multipliers (len>=horizon)

        # Effective root depth (after multiplier) & FAO-56 constants (computed)
        self.Zr_eff_m: float = float(cfg.Zr_m)
        self.TAW: float = 0.0
        self.RAW: float = 0.0
        self._recompute_soil_params()

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

        # last action (clipped) and raw action (for clip_rate)
        self.last_I: float = 0.0
        self.last_I_raw: float = 0.0
        self.last_I_clipped: int = 0

        # cached daily variables (debug/paper)
        self.last_ET0: float = 0.0
        self.last_ETc: float = 0.0
        self.last_Kc: float = 0.0
        self.last_Ks: float = 0.0
        self.last_DP: float = 0.0

    # -----------------------
    # Domain perturbation API
    # -----------------------
    def _recompute_soil_params(self) -> None:
        """Recompute TAW/RAW and derived quantities after Zr multiplier changes."""
        zr_mult = float(self.domain_params.get("Zr_mult", 1.0))
        # clamp to avoid degenerate depth
        self.Zr_eff_m = float(np.clip(self.cfg.Zr_m * zr_mult, 0.05, 5.0))
        self.TAW = float(calc_TAW(self.cfg.theta_fc, self.cfg.theta_wp, self.Zr_eff_m))
        self.RAW = float(calc_RAW(self.cfg.p, self.TAW))

    def set_domain_params(
            self,
            *,
            ET0_mult: Optional[float] = None,
            Kc_mult: Optional[float] = None,
            Zr_mult: Optional[float] = None,
            ET0_mult_series: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set episode-level domain multipliers.

        - ET0_mult: scalar multiplier applied to ET0 each day (unless series is set)
        - ET0_mult_series: per-day multiplier sequence; if provided, overrides ET0_mult
        - Kc_mult: scalar multiplier applied to Kc schedule
        - Zr_mult: scalar multiplier applied to root depth, affecting TAW/RAW and theta mapping
        """
        if ET0_mult is not None:
            self.domain_params["ET0_mult"] = float(ET0_mult)
        if Kc_mult is not None:
            self.domain_params["Kc_mult"] = float(Kc_mult)
        if Zr_mult is not None:
            self.domain_params["Zr_mult"] = float(Zr_mult)

        if ET0_mult_series is not None:
            arr = np.asarray(ET0_mult_series, dtype=np.float32).reshape(-1)
            self._ET0_mult_series = arr
        else:
            self._ET0_mult_series = None

        # Zr affects TAW/RAW and theta mapping
        self._recompute_soil_params()

        # update observation bounds (TAW may change)
        try:
            high = self.observation_space.high.copy()
            high[0] = max(self.TAW, 1.0)
            self.observation_space = spaces.Box(
                low=self.observation_space.low, high=high, dtype=np.float32
            )
        except Exception:
            # If something goes wrong, keep the original space (non-critical).
            pass

    def _et0_mult_today(self) -> float:
        if self._ET0_mult_series is not None and self._ET0_mult_series.size > 0:
            idx = int(np.clip(self.day, 0, self._ET0_mult_series.size - 1))
            return float(self._ET0_mult_series[idx])
        return float(self.domain_params.get("ET0_mult", 1.0))

    # -----------------------
    # Core helpers
    # -----------------------
    def _safe_ET0(self, w: Dict[str, Any]) -> float:
        """
        ET0 calculator:
        - Prefer PM (Penman–Monteith) if inputs exist
        - Fallback if missing keys / errors
        - Apply optional ET0 multiplier (robustness / domain rand)

        calc_ET0_PM 内部也可以做 fallback，但这里加一层防御更稳健。
        """
        try:
            et0 = float(calc_ET0_PM(w))
            if not np.isfinite(et0) or et0 < 0.0:
                raise ValueError("ET0 invalid")
        except Exception:
            et0 = float(calc_ET0_fallback(w))
            if not np.isfinite(et0) or et0 < 0.0:
                et0 = 0.0

        et0 *= self._et0_mult_today()
        if not np.isfinite(et0) or et0 < 0.0:
            et0 = 0.0
        return float(et0)

    def _get_obs(self) -> np.ndarray:
        # Next-state observation uses current self.day (already updated)
        w = self.weather.get_day(self.day)
        ET0 = self._safe_ET0(w)
        theta = float(Dr_to_theta(self.Dr, self.cfg.theta_fc, self.Zr_eff_m))
        s = stage_norm(self.cfg, self.day)
        return np.array([self.Dr, theta, ET0, s], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        theta = float(Dr_to_theta(self.Dr, self.cfg.theta_fc, self.Zr_eff_m))
        s = stage_norm(self.cfg, self.day)

        return {
            "day": int(self.day),
            "stage_norm": float(s),  # ✅ RewardWrapper dynamic target uses it
            "Dr_mm": float(self.Dr),
            "theta": float(theta),
            "TAW_mm": float(self.TAW),
            "RAW_mm": float(self.RAW),
            "Zr_eff_m": float(self.Zr_eff_m),
            "ET0_mult": float(self._et0_mult_today()),
            "Kc_mult": float(self.domain_params.get("Kc_mult", 1.0)),
            "Zr_mult": float(self.domain_params.get("Zr_mult", 1.0)),
            # cached daily vars (after step they are updated)
            "ET0": float(self.last_ET0),
            "ETc": float(self.last_ETc),
            "Kc": float(self.last_Kc),
            "Ks": float(self.last_Ks),
            "DP": float(self.last_DP),
            # action logging
            "I_mm": float(self.last_I),
            "I_raw_mm": float(self.last_I_raw),
            "clipped": int(self.last_I_clipped),
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
        # Note: if Zr_mult changes TAW, Dr init follows updated TAW.
        self.Dr = float(self.rng.uniform(0.2 * self.TAW, 0.6 * self.TAW))
        self.prev_Dr = self.Dr

        # action logs
        self.last_I = 0.0
        self.last_I_raw = 0.0
        self.last_I_clipped = 0

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
        - Apply irrigation action (with clipping)
        - Update Dr using FAO-56-like water balance
        - Advance day
        """
        # --- robust action parsing ---
        if isinstance(action, (float, int)):
            a = float(action)
        else:
            a = float(np.asarray(action).reshape(-1)[0])

        I_raw = float(a)
        I = float(np.clip(I_raw, 0.0, self.cfg.a_max_mm))
        self.last_I_raw = I_raw
        self.last_I_clipped = int(abs(I - I_raw) > 1e-9)

        # current day weather
        w = self.weather.get_day(self.day)

        # ET0 / crop / stress
        ET0 = self._safe_ET0(w)
        Kc_stage = float(kc_by_stage(self.cfg, self.day))
        Kc = float(Kc_stage * float(self.domain_params.get("Kc_mult", 1.0)))
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
        theta_next = float(Dr_to_theta(self.Dr, self.cfg.theta_fc, self.Zr_eff_m))

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
