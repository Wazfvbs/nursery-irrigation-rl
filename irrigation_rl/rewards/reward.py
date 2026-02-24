from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math

def distance_to_interval(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0

@dataclass
class RewardConfig:
    w_track: float = 1.0
    w_water: float = 0.05
    w_improve: float = 0.2
    w_smooth: float = 0.02
    w_safe: float = 5.0
    w_ucb: float = 0.0  # set >0 when UCB enabled

class RewardFunction:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.prev_err: float | None = None
        self.prev_I: float = 0.0

    def reset(self):
        self.prev_err = None
        self.prev_I = 0.0

    def compute(
        self,
        Dr: float,
        Dr_lo: float,
        Dr_hi: float,
        I: float,
        theta: float,
        theta_wp: float,
        ucb_bonus: float = 0.0,
        unsafe: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        err = distance_to_interval(Dr, Dr_lo, Dr_hi)

        r_track = -self.cfg.w_track * err
        r_water = -self.cfg.w_water * I
        r_smooth = -self.cfg.w_smooth * abs(I - self.prev_I)

        r_improve = 0.0
        if self.prev_err is not None:
            r_improve = self.cfg.w_improve * (self.prev_err - err)

        r_safe = 0.0
        if theta < theta_wp or unsafe:
            r_safe = -self.cfg.w_safe

        r_ucb = self.cfg.w_ucb * ucb_bonus

        reward = r_track + r_water + r_smooth + r_improve + r_safe + r_ucb

        self.prev_err = err
        self.prev_I = I

        terms = {
            "err": float(err),
            "r_track": float(r_track),
            "r_water": float(r_water),
            "r_smooth": float(r_smooth),
            "r_improve": float(r_improve),
            "r_safe": float(r_safe),
            "r_ucb": float(r_ucb),
            "reward": float(reward),
        }
        return float(reward), terms
