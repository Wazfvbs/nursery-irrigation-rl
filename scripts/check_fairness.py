from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def find_metrics_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "metrics.json":
                out.append(os.path.join(dirpath, fn))
    return out


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_file(
        metrics_path: str,
        *,
        i_max: float,
        tol: float,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    try:
        m = load_json(metrics_path)
    except Exception as e:
        return [f"{metrics_path}: cannot read metrics.json ({e})"], warns

    for k in ("method", "seed", "trajectory_csv"):
        if k not in m:
            errors.append(f"{metrics_path}: missing key '{k}'")

    if "clip_rate" in m:
        cr = float(m["clip_rate"])
        if not (0.0 <= cr <= 1.0):
            errors.append(f"{metrics_path}: clip_rate out of range [0,1], got {cr}")
    else:
        warns.append(f"{metrics_path}: clip_rate missing")

    if "ref_source" in m and str(m["ref_source"]) != "ref":
        errors.append(f"{metrics_path}: ref_source must be 'ref', got {m['ref_source']}")

    traj_csv = m.get("trajectory_csv")
    if not isinstance(traj_csv, str) or not os.path.exists(traj_csv):
        errors.append(f"{metrics_path}: trajectory_csv not found ({traj_csv})")
        return errors, warns

    try:
        df = pd.read_csv(traj_csv)
    except Exception as e:
        errors.append(f"{metrics_path}: cannot read trajectory_csv ({e})")
        return errors, warns

    required_cols = ["Dr_lo_ref", "Dr_hi_ref", "Dr_mid_ref", "I", "I_raw", "clipped"]
    for c in required_cols:
        if c not in df.columns:
            errors.append(f"{metrics_path}: trajectory missing column '{c}'")

    if "I" in df.columns:
        I = pd.to_numeric(df["I"], errors="coerce").to_numpy(dtype=float)
        if np.isnan(I).any():
            errors.append(f"{metrics_path}: NaN in I")
        if np.any(I < -tol) or np.any(I > i_max + tol):
            errors.append(f"{metrics_path}: I outside [0, {i_max}]")

    if ("I" in df.columns) and ("I_raw" in df.columns) and ("clipped" in df.columns):
        I = pd.to_numeric(df["I"], errors="coerce").to_numpy(dtype=float)
        I_raw = pd.to_numeric(df["I_raw"], errors="coerce").to_numpy(dtype=float)
        clipped = pd.to_numeric(df["clipped"], errors="coerce").to_numpy(dtype=float)
        inferred = (np.abs(I - I_raw) > tol).astype(int)
        observed = (clipped > 0.5).astype(int)
        if inferred.shape == observed.shape:
            mismatch = int(np.sum(inferred != observed))
            if mismatch > 0:
                errors.append(f"{metrics_path}: clipped flag mismatch count={mismatch}")

    if "ucb_bonus" in df.columns:
        ucb = pd.to_numeric(df["ucb_bonus"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.any(np.abs(ucb) > tol):
            errors.append(f"{metrics_path}: non-zero ucb_bonus during evaluation")
    else:
        warns.append(f"{metrics_path}: ucb_bonus column missing (cannot verify eval UCB=0)")

    return errors, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs")
    ap.add_argument("--i_max", type=float, default=15.0)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--dev_start", type=int, default=32)
    ap.add_argument("--dev_num", type=int, default=10)
    ap.add_argument("--report_start", type=int, default=42)
    ap.add_argument("--report_num", type=int, default=10)
    ap.add_argument("--strict_report_only", action="store_true", help="require all scanned seeds in report set")
    args = ap.parse_args()

    metric_files = find_metrics_files(args.root)
    if not metric_files:
        print(f"[WARN] no metrics.json found under: {args.root}")
        return

    dev_set = set(range(int(args.dev_start), int(args.dev_start) + int(args.dev_num)))
    report_set = set(range(int(args.report_start), int(args.report_start) + int(args.report_num)))
    overlap = dev_set & report_set
    if overlap:
        raise SystemExit(f"[FAIL] dev/report seed sets overlap: {sorted(overlap)}")

    all_errors: List[str] = []
    all_warns: List[str] = []
    all_seeds: List[int] = []

    for mp in metric_files:
        try:
            seed_val = load_json(mp).get("seed")
            if seed_val is not None:
                all_seeds.append(int(seed_val))
        except Exception:
            pass
        errs, warns = check_file(mp, i_max=float(args.i_max), tol=float(args.tol))
        all_errors.extend(errs)
        all_warns.extend(warns)

    known_set = dev_set | report_set
    for s in sorted(set(all_seeds)):
        if s not in known_set:
            all_warns.append(f"seed {s} is outside dev/report ranges")
        if args.strict_report_only and s not in report_set:
            all_errors.append(f"seed {s} is not in report set")

    print(f"[INFO] checked files: {len(metric_files)}")
    print(f"[INFO] warnings: {len(all_warns)}")
    for w in all_warns[:50]:
        print(f"[WARN] {w}")
    if len(all_warns) > 50:
        print(f"[WARN] ... {len(all_warns) - 50} more warnings")

    if all_errors:
        print(f"[FAIL] errors: {len(all_errors)}")
        for e in all_errors[:100]:
            print(f"[ERR] {e}")
        if len(all_errors) > 100:
            print(f"[ERR] ... {len(all_errors) - 100} more errors")
        raise SystemExit(1)

    print("[OK] fairness checks passed")


if __name__ == "__main__":
    main()
