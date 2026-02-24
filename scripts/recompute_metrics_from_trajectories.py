from __future__ import annotations

"""Recompute metrics.json from existing trajectory.csv files.

Why this script
---------------
After you change metric definitions (e.g., switch to reference-interval metrics),
you typically *do not* need to retrain models. As long as you have
`trajectory.csv`, you can recompute metrics and regenerate tables.

Usage
-----
python scripts/recompute_metrics_from_trajectories.py --root outputs
"""

import argparse
import json
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from typing import Any, Dict, List

from irrigation_rl.train.metrics import compute_metrics_from_csv


def find_files(root: str, filename: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            out.append(os.path.join(dirpath, filename))
    return out


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: str, obj: Dict[str, Any]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs", help="scan this directory")
    ap.add_argument("--dry", action="store_true", help="print changes only, do not write")
    args = ap.parse_args()

    metric_paths = find_files(args.root, "metrics.json")
    if not metric_paths:
        print("[WARN] no metrics.json found under:", args.root)
        return

    updated = 0
    skipped = 0

    for mp in metric_paths:
        try:
            j = load_json(mp)
            traj = j.get("trajectory_csv")
            if not isinstance(traj, str) or not os.path.exists(traj):
                # try local fallback
                guess = os.path.join(os.path.dirname(mp), "eval", "trajectory.csv")
                if os.path.exists(guess):
                    traj = guess
                else:
                    skipped += 1
                    continue

            m = compute_metrics_from_csv(traj)

            # preserve identifiers / metadata
            keep_keys = [
                "method", "seed", "case", "scenario", "model_path",
                "eval_out", "out_dir", "noise",
            ]
            new_j: Dict[str, Any] = {k: j.get(k) for k in keep_keys if k in j}
            new_j.update(m)
            new_j["trajectory_csv"] = traj

            if args.dry:
                print("[DRY]", mp)
            else:
                save_json(mp, new_j)
            updated += 1
        except Exception as e:
            skipped += 1
            continue

    print(f"[OK] recompute finished. updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    main()
