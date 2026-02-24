from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TargetConfig:
    # fixed target（用于 w/o_Target）
    low_frac_of_TAW: float = 0.20
    high_frac_of_RAW: float = 0.90

    # dynamic target（用于 Full）
    # 三阶段：early / mid / late
    early_low_frac_TAW: float = 0.15
    early_high_frac_RAW: float = 0.70

    mid_low_frac_TAW: float = 0.20
    mid_high_frac_RAW: float = 0.90

    late_low_frac_TAW: float = 0.25
    late_high_frac_RAW: float = 1.00

class DynamicTarget:
    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg

    def get_interval(self, TAW: float, RAW: float, stage_norm: float = 0.5):
        # stage_norm ∈ [0,1]
        if stage_norm < 0.33:
            lo = self.cfg.early_low_frac_TAW * TAW
            hi = self.cfg.early_high_frac_RAW * RAW
        elif stage_norm < 0.66:
            lo = self.cfg.mid_low_frac_TAW * TAW
            hi = self.cfg.mid_high_frac_RAW * RAW
        else:
            lo = self.cfg.late_low_frac_TAW * TAW
            hi = self.cfg.late_high_frac_RAW * RAW

        if hi < lo:
            hi = lo
        return float(lo), float(hi)

