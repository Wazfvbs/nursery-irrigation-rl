from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

class WeatherProvider:
    def reset(self, seed: int | None = None) -> None:
        pass

    def get_day(self, t: int) -> Dict[str, float]:
        raise NotImplementedError

@dataclass
class AssumptionWeatherConfig:
    T_mean_C: float = 20.0
    RH_pct: float = 60.0
    u2_mps: float = 1.0
    Rs_MJ_m2_day: float = 15.0
    noise_sigma: float = 0.0

class AssumptionWeatherProvider(WeatherProvider):
    def __init__(self, cfg: AssumptionWeatherConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed if seed is not None else 0)

    def get_day(self, t: int) -> Dict[str, float]:
        n = self.cfg.noise_sigma
        return {
            "T_mean_C": float(self.cfg.T_mean_C + self.rng.normal(0.0, n)),
            "RH_pct": float(self.cfg.RH_pct + self.rng.normal(0.0, n)),
            "u2_mps": float(max(0.0, self.cfg.u2_mps + self.rng.normal(0.0, n))),
            "Rs_MJ_m2_day": float(max(0.0, self.cfg.Rs_MJ_m2_day + self.rng.normal(0.0, n))),
        }

class ExternalCSVWeatherProvider(WeatherProvider):
    """
    Expects a CSV with columns (any subset is okay, missing handled downstream):
      day, T_mean_C, RH_pct, u2_mps, Rs_MJ_m2_day
    """
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def reset(self, seed: int | None = None) -> None:
        return

    def get_day(self, t: int) -> Dict[str, float]:
        idx = int(t) % len(self.df)
        row = self.df.iloc[idx].to_dict()
        # Normalize keys
        return {
            "T_mean_C": float(row.get("T_mean_C", row.get("T", 20.0))),
            "RH_pct": float(row.get("RH_pct", row.get("RH", 60.0))),
            "u2_mps": float(row.get("u2_mps", row.get("u2", 1.0))),
            "Rs_MJ_m2_day": float(row.get("Rs_MJ_m2_day", row.get("Rs", 15.0))),
        }
