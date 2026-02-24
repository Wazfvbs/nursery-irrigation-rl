from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Optional

@dataclass
class FAOParams:
    theta_fc: float
    theta_wp: float
    Zr_m: float
    p: float

def calc_TAW(theta_fc: float, theta_wp: float, Zr_m: float) -> float:
    """Total available water (mm)."""
    return 1000.0 * max(theta_fc - theta_wp, 0.0) * max(Zr_m, 0.0)

def calc_RAW(p: float, TAW: float) -> float:
    """Readily available water (mm)."""
    return max(p, 0.0) * max(TAW, 0.0)

def calc_Ks(Dr: float, RAW: float, TAW: float) -> float:
    """Soil water stress coefficient Ks (dimensionless)."""
    Dr = max(Dr, 0.0)
    if Dr <= RAW:
        return 1.0
    denom = max(TAW - RAW, 1e-8)
    return max(min((TAW - Dr) / denom, 1.0), 0.0)

def theta_to_Dr(theta: float, theta_fc: float, Zr_m: float) -> float:
    """Map volumetric water content to root-zone depletion Dr (mm)."""
    theta = max(theta, 0.0)
    return 1000.0 * max(theta_fc - theta, 0.0) * max(Zr_m, 0.0)

def Dr_to_theta(Dr: float, theta_fc: float, Zr_m: float) -> float:
    """Map root-zone depletion Dr (mm) to volumetric water content theta (m3/m3)."""
    Zr_m = max(Zr_m, 1e-8)
    return max(theta_fc - Dr / (1000.0 * Zr_m), 0.0)

def calc_DP_from_theta(theta: float, theta_fc: float, Zr_m: float) -> float:
    """
    Minimal deep percolation model (mm/day).
    If theta > theta_fc, excessive water is drained out in the same day.
    This is a minimal placeholder; can be refined later.
    """
    if theta <= theta_fc:
        return 0.0
    excess = (theta - theta_fc) * 1000.0 * max(Zr_m, 0.0)
    return max(excess, 0.0)

# --- ET0: Penman-Monteith (placeholder) ---
def calc_ET0_PM(weather: Dict[str, float]) -> float:
    """
    Reference ET0 (mm/day) using FAO-56 Penman鈥揗onteith.
    This is a placeholder to be implemented with full variables:
      T_mean, RH, u2, Rn, G, pressure, etc.
    If inputs are limited, use fallback approximation.
    """
    return calc_ET0_fallback(weather)

def calc_ET0_fallback(weather: Dict[str, float]) -> float:
    """
    Fallback ET0 (mm/day) approximation used when sensor-limited.
    You can replace with Hargreaves or other simplified method.
    """
    T = float(weather.get("T_mean_C", 20.0))
    Rs = float(weather.get("Rs_MJ_m2_day", 15.0))
    # Very simple proxy: scaled radiation with temperature factor
    return max(0.0, 0.0023 * (T + 17.8) * Rs)
