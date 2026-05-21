from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple


def distance_to_interval(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0


@dataclass
class RewardConfig:
    # method_design.md initial weights.
    w_track: float = 1.0
    w_water: float = 0.05
    w_stress: float = 2.0
    w_over: float = 1.0
    w_smooth: float = 0.05
    w_safe: float = 3.0
    w_ucb: float = 0.10
    w_uncertainty: float = 0.10

    # Safety violation threshold for deep percolation.
    dp_max: float = 0.0


class RewardFunction:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.prev_I: float = 0.0

    def reset(self) -> None:
        self.prev_I = 0.0

    def compute(
        self,
        *,
        Dr: float,
        Dr_prev: float,
        Dr_lo: float,
        Dr_hi: float,
        I: float,
        I_prev: float,
        Imax: float,
        TAW: float,
        RAW: float,
        DP: Optional[float] = None,
        P: float = 0.0,
        ucb_bonus: float = 0.0,
        uncertainty: float = 0.0,
        c_uncertain: float = 0.0,
        unsafe: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        taw = max(float(TAW), 1e-8)
        imax = max(float(Imax), 1e-8)
        Dr = float(Dr)
        Dr_prev = float(Dr_prev)
        I = float(I)
        I_prev = float(I_prev)
        RAW = float(RAW)
        dp_missing = DP is None
        if dp_missing:
            dp_mm = None
        else:
            try:
                dp_value = float(DP)
                dp_mm = max(0.0, dp_value) if math.isfinite(dp_value) else None
            except (TypeError, ValueError):
                dp_mm = None
            dp_missing = dp_mm is None
        P = max(0.0, float(P))

        e_target_mm = distance_to_interval(Dr, float(Dr_lo), float(Dr_hi))
        e_target = e_target_mm / taw
        water_use = I / imax
        stress = max(0.0, Dr - RAW) / taw

        over_mm = max(0.0, I + P - Dr_prev) if dp_missing else float(dp_mm)
        over_irrigation = over_mm / taw
        smoothness = abs(I - I_prev) / imax

        safety_violation = 0.0
        if unsafe:
            safety_violation += 1.0
        if Dr >= taw:
            safety_violation += 1.0
        if over_mm > float(self.cfg.dp_max):
            safety_violation += 1.0

        r_track = -self.cfg.w_track * e_target
        r_water = -self.cfg.w_water * water_use
        r_stress = -self.cfg.w_stress * stress
        r_over = -self.cfg.w_over * over_irrigation
        r_smooth = -self.cfg.w_smooth * smoothness
        r_safe = -self.cfg.w_safe * safety_violation
        r_ucb = self.cfg.w_ucb * float(ucb_bonus)
        r_uncertainty = -self.cfg.w_uncertainty * float(c_uncertain)

        reward = (
            r_track
            + r_water
            + r_stress
            + r_over
            + r_smooth
            + r_safe
            + r_ucb
            + r_uncertainty
        )

        self.prev_I = I

        terms = {
            "e_target": float(e_target),
            "e_target_mm": float(e_target_mm),
            "water_use": float(water_use),
            "stress": float(stress),
            "over_irrigation": float(over_irrigation),
            "over_irrigation_mm": float(over_mm),
            "smoothness": float(smoothness),
            "safety_violation": float(safety_violation),
            "ucb_bonus": float(ucb_bonus),
            "uncertainty": float(uncertainty),
            "c_uncertain": float(c_uncertain),
            "r_track": float(r_track),
            "r_water": float(r_water),
            "r_stress": float(r_stress),
            "r_over": float(r_over),
            "r_smooth": float(r_smooth),
            "r_safe": float(r_safe),
            "r_ucb": float(r_ucb),
            "r_uncertainty": float(r_uncertainty),
            "reward": float(reward),
        }
        return float(reward), terms
