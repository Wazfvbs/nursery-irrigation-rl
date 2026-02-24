from __future__ import annotations

"""Metrics utilities.

Goal
----
We evaluate *all* methods (PPO, baselines, robustness, ablations) using the
same **reference target interval** (the Full/dynamic target). This avoids
"self-defined exam" issues in ablations (e.g., w/o_Target achieving 100% by
changing its own target).

All exported metrics are computed from `trajectory.csv` produced by
`irrigation_rl.train.evaluate.evaluate_policy` (or baseline rollouts).

Core metrics (paper main tables)
-------------------------------
* TIR_ref           : Time-in-Range w.r.t. reference interval
* IAE_mid_ref       : Integral absolute error to reference midpoint
* TotalIrrigation_mm: Total irrigation amount
* StressDays_ref    : Days above reference upper bound
* ActionTV          : Total variation of action sequence

Recommended extras (reproducibility / interpretation)
-----------------------------------------------------
* UnderDays_ref     : Days below reference lower bound
* clip_rate         : Fraction of steps where action was clipped to [0, I_max]

Supplement metrics
------------------
* MAE_ref_mm, RMSE_ref_mm, ActionStd, RewardMean, RewardSum
"""

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


# -------------------------
# Low-level helpers
# -------------------------

def _interval_outside_error(Dr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance to interval [lo, hi], 0 if inside."""
    err = np.zeros_like(Dr, dtype=float)
    m1 = Dr < lo
    m2 = Dr > hi
    err[m1] = lo[m1] - Dr[m1]
    err[m2] = Dr[m2] - hi[m2]
    return err


def _action_tv(I: np.ndarray) -> float:
    if I.size <= 1:
        return 0.0
    return float(np.sum(np.abs(I[1:] - I[:-1])))


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else float("nan")


def _safe_sum(x: np.ndarray) -> float:
    return float(np.sum(x)) if x.size > 0 else 0.0


def _to_float_array(series: pd.Series) -> np.ndarray:
    """Convert a pandas Series to float array robustly."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _compute_clip_rate(df: pd.DataFrame) -> float:
    """
    Compute clip_rate from trajectory dataframe.

    Priority:
    1) Use 'clipped' column if present (0/1 int)
    2) Fallback: infer from |I - I_raw| if both present
    """
    if "clipped" in df.columns:
        clipped = _to_float_array(df["clipped"])
        clipped = np.nan_to_num(clipped, nan=0.0)
        if clipped.size == 0:
            return float("nan")
        # treat >0.5 as clipped
        return float(np.mean(clipped > 0.5))

    if ("I" in df.columns) and ("I_raw" in df.columns):
        I = _to_float_array(df["I"])
        I_raw = _to_float_array(df["I_raw"])
        if I.size == 0:
            return float("nan")
        return float(np.mean(np.abs(I - I_raw) > 1e-9))

    return float("nan")


# -------------------------
# Public API
# -------------------------

def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute ref-based metrics from a trajectory dataframe."""
    required = ["Dr", "I"]
    for c in required:
        if c not in df.columns:
            return {"error": "missing_cols", "missing": required}

    Dr = _to_float_array(df["Dr"])
    I = _to_float_array(df["I"])

    # reference interval preferred
    if ("Dr_lo_ref" in df.columns) and ("Dr_hi_ref" in df.columns):
        lo_ref = _to_float_array(df["Dr_lo_ref"])
        hi_ref = _to_float_array(df["Dr_hi_ref"])
        ref_source = "ref"
    elif ("Dr_lo" in df.columns) and ("Dr_hi" in df.columns):
        # fallback for legacy trajectories
        lo_ref = _to_float_array(df["Dr_lo"])
        hi_ref = _to_float_array(df["Dr_hi"])
        ref_source = "train_fallback"
    else:
        return {"error": "missing_cols", "missing": ["Dr_lo_ref/Dr_hi_ref or Dr_lo/Dr_hi"]}

    if "Dr_mid_ref" in df.columns:
        mid_ref = _to_float_array(df["Dr_mid_ref"])
    else:
        mid_ref = 0.5 * (lo_ref + hi_ref)

    # --- core ---
    in_ref = (Dr >= lo_ref) & (Dr <= hi_ref)
    TIR_ref = float(np.mean(in_ref)) if Dr.size > 0 else float("nan")
    IAE_mid_ref = float(np.sum(np.abs(Dr - mid_ref))) if Dr.size > 0 else float("nan")
    TotalIrrigation_mm = _safe_sum(I)
    StressDays_ref = int(np.sum(Dr > hi_ref))
    ActionTV = _action_tv(I)

    # --- recommended extras ---
    UnderDays_ref = int(np.sum(Dr < lo_ref))
    clip_rate = _compute_clip_rate(df)

    # --- supplementary ---
    err_out = _interval_outside_error(Dr, lo_ref, hi_ref)
    MAE_ref_mm = _safe_mean(np.abs(err_out))
    RMSE_ref_mm = float(np.sqrt(_safe_mean(err_out ** 2))) if Dr.size > 0 else float("nan")
    ActionStd = float(np.std(I)) if I.size > 0 else float("nan")

    # optional reward stats
    RewardMean = float(np.mean(_to_float_array(df["reward"]))) if "reward" in df.columns else float("nan")
    RewardSum = float(np.sum(_to_float_array(df["reward"]))) if "reward" in df.columns else float("nan")

    out: Dict[str, Any] = {
        "N_steps": int(len(df)),
        "ref_source": ref_source,

        # core
        "TIR_ref": TIR_ref,
        "IAE_mid_ref": IAE_mid_ref,
        "TotalIrrigation_mm": TotalIrrigation_mm,
        "StressDays_ref": StressDays_ref,
        "ActionTV": ActionTV,

        # recommended extras
        "UnderDays_ref": UnderDays_ref,
        "clip_rate": clip_rate,

        # supp
        "MAE_ref_mm": MAE_ref_mm,
        "RMSE_ref_mm": RMSE_ref_mm,
        "ActionStd": ActionStd,
        "RewardMean": RewardMean,
        "RewardSum": RewardSum,

        # alias (helps table scripts / human-readable mapping)
        "TotalIrrigation": TotalIrrigation_mm,
    }
    return out


def compute_metrics_from_csv(csv_path: str) -> Dict[str, Any]:
    """Read trajectory.csv then compute metrics."""
    df = pd.read_csv(csv_path)
    m = compute_metrics_from_df(df)
    m["trajectory_csv"] = csv_path
    return m


# Backward compatible name (some scripts used compute_metrics_from_csv inline)
compute_metrics_from_trajectory_csv = compute_metrics_from_csv


def compute_metrics(traj: Dict[str, Iterable[Any]]) -> Dict[str, Any]:
    """Legacy dict-based API (kept for backward compatibility)."""
    df = pd.DataFrame({k: list(v) for k, v in traj.items()})
    return compute_metrics_from_df(df)
