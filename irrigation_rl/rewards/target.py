from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TargetConfig:
    # Fixed interval for w/o_Target ablation.
    fixed_low_frac_TAW: float = 0.15
    fixed_high_frac_TAW: float = 0.35

    # Dynamic stage-based intervals from method_design.md.
    early_low_frac_TAW: float = 0.20
    early_high_frac_TAW: float = 0.40
    mid_low_frac_TAW: float = 0.25
    mid_high_frac_TAW: float = 0.50
    late_low_frac_TAW: float = 0.30
    late_high_frac_TAW: float = 0.60

    # ET0-based upper-bound correction.
    et0_mean: float = 0.0
    lambda_et: float = 0.10
    min_width: float = 5.0


class DynamicTarget:
    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg

    def get_interval(self, *, stage_norm: float, et0: float, taw: float) -> Tuple[float, float]:
        taw = max(float(taw), 1e-8)
        s = float(stage_norm)

        if s < 0.33:
            lo = self.cfg.early_low_frac_TAW * taw
            hi = self.cfg.early_high_frac_TAW * taw
        elif s < 0.66:
            lo = self.cfg.mid_low_frac_TAW * taw
            hi = self.cfg.mid_high_frac_TAW * taw
        else:
            lo = self.cfg.late_low_frac_TAW * taw
            hi = self.cfg.late_high_frac_TAW * taw

        et0_excess = max(0.0, float(et0) - float(self.cfg.et0_mean))
        hi = hi - float(self.cfg.lambda_et) * et0_excess
        hi = max(hi, lo + float(self.cfg.min_width))
        hi = min(hi, taw)
        if hi < lo:
            hi = lo
        return float(lo), float(hi)

