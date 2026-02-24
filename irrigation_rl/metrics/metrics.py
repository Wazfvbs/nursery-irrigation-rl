from __future__ import annotations
from typing import Dict, List
import numpy as np

def compute_metrics(traj: Dict[str, List[float]]) -> Dict[str, float]:
    """
    traj keys recommended:
      Dr, Dr_lo, Dr_hi, I, terminated_flag
    """
    Dr = np.asarray(traj.get("Dr", []), dtype=float)
    Dr_lo = np.asarray(traj.get("Dr_lo", []), dtype=float)
    Dr_hi = np.asarray(traj.get("Dr_hi", []), dtype=float)
    I = np.asarray(traj.get("I", []), dtype=float)

    if len(Dr) == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "TotalIrrigation": 0.0}

    # Distance to interval
    err = np.zeros_like(Dr)
    err[Dr < Dr_lo] = Dr_lo[Dr < Dr_lo] - Dr[Dr < Dr_lo]
    err[Dr > Dr_hi] = Dr[Dr > Dr_hi] - Dr_hi[Dr > Dr_hi]

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    totalI = float(np.sum(I))

    stress_days = float(np.sum(Dr > Dr_hi))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "TotalIrrigation": totalI,
        "StressDays": stress_days,
    }
