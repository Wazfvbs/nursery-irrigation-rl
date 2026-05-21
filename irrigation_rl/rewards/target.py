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

    # Stage boundaries in the same normalized day coordinate as stage_norm.
    # build_env derives these from env.yaml, e.g. 20/90 and 70/90 for 20/50/20.
    early_end_norm: float = 1.0 / 3.0
    mid_end_norm: float = 2.0 / 3.0


class DynamicTarget:
    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg

    def get_interval(self, *, stage_norm: float, et0: float, taw: float) -> Tuple[float, float]:
        taw = max(float(taw), 1e-8)
        s = float(stage_norm)
        early_end = max(0.0, min(float(self.cfg.early_end_norm), 1.0))
        mid_end = max(early_end, min(float(self.cfg.mid_end_norm), 1.0))

        if s < early_end:
            lo = self.cfg.early_low_frac_TAW * taw
            hi = self.cfg.early_high_frac_TAW * taw
        elif s < mid_end:
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
