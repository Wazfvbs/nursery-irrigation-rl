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

def _first_float(weather: Dict[str, float], *keys: str, default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key not in weather:
            continue
        try:
            value = float(weather[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def _sat_vapor_pressure_kpa(temp_c: float) -> float:
    return 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))


def _slope_vapor_pressure_curve(temp_c: float, es_kpa: float) -> float:
    return 4098.0 * es_kpa / ((temp_c + 237.3) ** 2)


def _pressure_from_elevation_kpa(elevation_m: float) -> float:
    return 101.3 * (((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26)


def _extraterrestrial_radiation_mj_m2_day(latitude_rad: float, day_of_year: int) -> float:
    gsc = 0.0820  # MJ m-2 min-1
    j = int(max(1, min(366, day_of_year)))
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * j / 365.0)
    solar_dec = 0.409 * math.sin(2.0 * math.pi * j / 365.0 - 1.39)
    x = -math.tan(latitude_rad) * math.tan(solar_dec)
    x = max(-1.0, min(1.0, x))
    ws = math.acos(x)
    return (
        (24.0 * 60.0 / math.pi)
        * gsc
        * dr
        * (
            ws * math.sin(latitude_rad) * math.sin(solar_dec)
            + math.cos(latitude_rad) * math.cos(solar_dec) * math.sin(ws)
        )
    )


def _net_radiation_from_solar(
    weather: Dict[str, float],
    *,
    temp_c: float,
    ea_kpa: float,
    rs_mj_m2_day: float,
    elevation_m: float,
) -> float:
    albedo = _first_float(weather, "albedo", "albedo_shortwave", default=0.23)
    albedo = 0.23 if albedo is None else max(0.0, min(float(albedo), 1.0))
    rns = (1.0 - albedo) * max(rs_mj_m2_day, 0.0)

    rso = _first_float(weather, "Rso_MJ_m2_day", "Rso")
    if rso is None:
        ra = _first_float(weather, "Ra_MJ_m2_day", "Ra")
        if ra is None:
            lat_rad = _first_float(weather, "latitude_rad", "lat_rad")
            lat_deg = _first_float(weather, "latitude_deg", "lat_deg", "lat")
            doy = _first_float(weather, "day_of_year", "doy", "DOY")
            if lat_rad is None and lat_deg is not None:
                lat_rad = math.radians(lat_deg)
            if lat_rad is not None and doy is not None:
                ra = _extraterrestrial_radiation_mj_m2_day(lat_rad, int(doy))
        if ra is not None:
            rso = (0.75 + 2e-5 * elevation_m) * max(ra, 0.0)

    if rso is None or rso <= 0.0:
        # Sensor-limited fallback for clear-sky radiation. This keeps the
        # Penman-Monteith pathway usable with T/RH/u2/Rs-only weather records.
        rso = max(rs_mj_m2_day / 0.70, rs_mj_m2_day, 1e-6)

    tmax = _first_float(weather, "T_max_C", "Tmax_C", "Tmax", "T_max", default=temp_c)
    tmin = _first_float(weather, "T_min_C", "Tmin_C", "Tmin", "T_min", default=temp_c)
    tmax_k = float(tmax) + 273.16
    tmin_k = float(tmin) + 273.16

    sigma = 4.903e-9  # Stefan-Boltzmann constant, MJ K-4 m-2 day-1
    cloudiness = 1.35 * max(0.0, min(rs_mj_m2_day / max(rso, 1e-8), 1.0)) - 0.35
    cloudiness = max(0.05, cloudiness)
    vapor_term = max(0.05, 0.34 - 0.14 * math.sqrt(max(ea_kpa, 0.0)))
    rnl = sigma * ((tmax_k ** 4 + tmin_k ** 4) / 2.0) * vapor_term * cloudiness
    return max(0.0, rns - max(0.0, rnl))


def calc_ET0_PM(weather: Dict[str, float]) -> float:
    """
    Reference ET0 (mm/day) using the daily FAO-56 Penman-Monteith equation.

    Required minimum inputs:
      T_mean_C, RH_pct, u2_mps, and either Rn_MJ_m2_day or Rs_MJ_m2_day.

    Optional inputs improve radiation/psychrometric terms:
      T_max_C, T_min_C, G_MJ_m2_day, pressure_kPa, elevation_m,
      latitude_deg/latitude_rad, day_of_year, Ra_MJ_m2_day, Rso_MJ_m2_day.

    If the minimum inputs are not available, this function falls back to the
    lightweight sensor-limited approximation used by earlier experiments.
    """
    T = _first_float(weather, "T_mean_C", "Tmean_C", "T", "temp_C")
    RH = _first_float(weather, "RH_pct", "RH", "relative_humidity")
    u2 = _first_float(weather, "u2_mps", "u2", "wind_speed_mps")

    if T is None or RH is None or u2 is None:
        return calc_ET0_fallback(weather)

    RH = max(0.0, min(float(RH), 100.0))
    u2 = max(0.0, float(u2))

    es = _sat_vapor_pressure_kpa(float(T))
    ea = es * RH / 100.0
    delta = _slope_vapor_pressure_curve(float(T), es)

    pressure = _first_float(weather, "pressure_kPa", "P_kPa", "atm_pressure_kPa")
    elevation = _first_float(weather, "elevation_m", "z_m", "altitude_m", default=0.0)
    if pressure is None:
        pressure = _pressure_from_elevation_kpa(float(elevation or 0.0))
    gamma = 0.000665 * float(pressure)

    rn = _first_float(weather, "Rn_MJ_m2_day", "Rn")
    if rn is None:
        rs = _first_float(weather, "Rs_MJ_m2_day", "Rs", "solar_radiation")
        if rs is None:
            return calc_ET0_fallback(weather)
        rn = _net_radiation_from_solar(
            weather,
            temp_c=float(T),
            ea_kpa=ea,
            rs_mj_m2_day=float(rs),
            elevation_m=float(elevation or 0.0),
        )

    G = _first_float(weather, "G_MJ_m2_day", "G", default=0.0)
    rn_minus_g = float(rn) - float(G or 0.0)

    numerator = (
        0.408 * delta * rn_minus_g
        + gamma * (900.0 / (float(T) + 273.0)) * u2 * (es - ea)
    )
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    if denominator <= 0.0:
        return calc_ET0_fallback(weather)
    return max(0.0, numerator / denominator)

def calc_ET0_fallback(weather: Dict[str, float]) -> float:
    """
    Fallback ET0 (mm/day) approximation used when sensor-limited.
    You can replace with Hargreaves or other simplified method.
    """
    T = float(weather.get("T_mean_C", 20.0))
    Rs = float(weather.get("Rs_MJ_m2_day", 15.0))
    # Very simple proxy: scaled radiation with temperature factor
    return max(0.0, 0.0023 * (T + 17.8) * Rs)
