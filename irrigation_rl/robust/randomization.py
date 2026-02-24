from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class RandomizationConfig:
    enabled: bool = True
    ET0_mult_min: float = 0.95
    ET0_mult_max: float = 1.05
    theta_sigma: float = 0.005
    Kc_mult_min: float = 0.95
    Kc_mult_max: float = 1.05
    Zr_mult_min: float = 0.95
    Zr_mult_max: float = 1.05

def apply_domain_randomization(params: Dict[str, float], cfg: RandomizationConfig, rng: np.random.Generator) -> Dict[str, float]:
    if not cfg.enabled:
        return dict(params)

    out = dict(params)
    # Example multipliers
    out["ET0_mult"] = float(rng.uniform(cfg.ET0_mult_min, cfg.ET0_mult_max))
    out["theta_sigma"] = float(cfg.theta_sigma)
    out["Kc_mult"] = float(rng.uniform(cfg.Kc_mult_min, cfg.Kc_mult_max))
    out["Zr_mult"] = float(rng.uniform(cfg.Zr_mult_min, cfg.Zr_mult_max))
    return out
