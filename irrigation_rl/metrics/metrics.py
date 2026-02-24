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

Supplement metrics
------------------
* MAE_ref_mm, RMSE_ref_mm, UnderDays_ref, ActionStd
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


# -------------------------
# Low-level helpers
# -------------------------

def _as_float_array(x: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


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


# -------------------------
# Public API
# -------------------------

def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute ref-based metrics from a trajectory dataframe."""
    required = ["Dr", "I"]
    for c in required:
        if c not in df.columns:
            return {"error": "missing_cols", "missing": required}

    Dr = df["Dr"].to_numpy(dtype=float)
    I = df["I"].to_numpy(dtype=float)

    # reference interval preferred
    if ("Dr_lo_ref" in df.columns) and ("Dr_hi_ref" in df.columns):
        lo_ref = df["Dr_lo_ref"].to_numpy(dtype=float)
        hi_ref = df["Dr_hi_ref"].to_numpy(dtype=float)
        ref_source = "ref"
    elif ("Dr_lo" in df.columns) and ("Dr_hi" in df.columns):
        # fallback for legacy trajectories
        lo_ref = df["Dr_lo"].to_numpy(dtype=float)
        hi_ref = df["Dr_hi"].to_numpy(dtype=float)
        ref_source = "train_fallback"
    else:
        return {"error": "missing_cols", "missing": ["Dr_lo_ref/Dr_hi_ref or Dr_lo/Dr_hi"]}

    if "Dr_mid_ref" in df.columns:
        mid_ref = df["Dr_mid_ref"].to_numpy(dtype=float)
    else:
        mid_ref = 0.5 * (lo_ref + hi_ref)

    # --- core ---
    in_ref = (Dr >= lo_ref) & (Dr <= hi_ref)
    TIR_ref = float(np.mean(in_ref)) if Dr.size > 0 else float("nan")
    IAE_mid_ref = float(np.sum(np.abs(Dr - mid_ref))) if Dr.size > 0 else float("nan")
    TotalIrrigation_mm = _safe_sum(I)
    StressDays_ref = int(np.sum(Dr > hi_ref))
    ActionTV = _action_tv(I)

    # --- supplementary ---
    err_out = _interval_outside_error(Dr, lo_ref, hi_ref)
    MAE_ref_mm = _safe_mean(np.abs(err_out))
    RMSE_ref_mm = float(np.sqrt(_safe_mean(err_out ** 2))) if Dr.size > 0 else float("nan")
    UnderDays_ref = int(np.sum(Dr < lo_ref))
    ActionStd = float(np.std(I)) if I.size > 0 else float("nan")

    # optional reward stats
    RewardMean = float(np.mean(df["reward"].to_numpy(dtype=float))) if "reward" in df.columns else float("nan")
    RewardSum = float(np.sum(df["reward"].to_numpy(dtype=float))) if "reward" in df.columns else float("nan")

    out: Dict[str, Any] = {
        "N_steps": int(len(df)),
        "ref_source": ref_source,
        # core
        "TIR_ref": TIR_ref,
        "IAE_mid_ref": IAE_mid_ref,
        "TotalIrrigation_mm": TotalIrrigation_mm,
        "StressDays_ref": StressDays_ref,
        "ActionTV": ActionTV,
        # supp
        "MAE_ref_mm": MAE_ref_mm,
        "RMSE_ref_mm": RMSE_ref_mm,
        "UnderDays_ref": UnderDays_ref,
        "ActionStd": ActionStd,
        # reward (not a primary metric)
        "RewardMean": RewardMean,
        "RewardSum": RewardSum,
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
