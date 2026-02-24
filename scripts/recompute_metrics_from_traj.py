# scripts/recompute_metrics_from_traj.py
import os
import json
import argparse
from pathlib import Path

import pandas as pd

# 按你补丁的实际位置/函数名调整：
from irrigation_rl.train.metrics import compute_metrics_from_csv  # <- 你补丁里统一计算指标的函数

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs", help="root folder to scan")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing metrics.json")
    args = ap.parse_args()

    root = Path(args.root)
    traj_paths = list(root.rglob("trajectory.csv"))
    if not traj_paths:
        print(f"[WARN] No trajectory.csv found under {root.resolve()}")
        return

    updated = 0
    skipped = 0

    for traj in traj_paths:
        run_dir = traj.parent
        metrics_path = run_dir / "metrics.json"

        # 若不覆盖且已有 metrics.json，就跳过
        if metrics_path.exists() and (not args.overwrite):
            # 如果它已经是新口径（含 TIR_ref），也可以跳过
            try:
                old = json.loads(metrics_path.read_text(encoding="utf-8"))
                if "TIR_ref" in old and "IAE_mid_ref" in old and "ActionTV" in old:
                    skipped += 1
                    continue
            except Exception:
                pass

        try:
            m = compute_metrics_from_csv(str(traj))  # 返回 dict
        except Exception as e:
            print(f"[FAIL] {traj}: {e}")
            continue

        # 写入/覆盖 metrics.json
        metrics_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        updated += 1

    print(f"[DONE] updated={updated}, skipped={skipped}, total_traj={len(traj_paths)}")

if __name__ == "__main__":
    main()
