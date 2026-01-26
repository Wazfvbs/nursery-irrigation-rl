from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
from typing import Dict, Any, List, Tuple

import pandas as pd


def find_metrics_json(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "metrics.json":
                paths.append(os.path.join(dirpath, fn))
    return paths


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def to_mean_std(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """
    输出 mean±std 的论文表格格式（数值列自动聚合）
    """
    numeric_cols = [
        "MAE_mm",
        "RMSE_mm",
        "TotalIrrigation_mm",
        "StressDays",
        "WithinTargetRatio",
        "ActionStd",
        "ActionTV",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    g = df.groupby(group_key)[numeric_cols].agg(["mean", "std"])
    # flatten columns
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()

    # 生成 mean±std 展示列（适合直接复制到论文表格）
    out = pd.DataFrame()
    out[group_key] = g[group_key]

    for col in numeric_cols:
        m = g[f"{col}_mean"]
        s = g[f"{col}_std"]
        out[col] = m.map(lambda x: f"{x:.6f}") + " ± " + s.map(lambda x: f"{x:.6f}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs", help="scan outputs/ for metrics.json")
    ap.add_argument("--out", type=str, default="outputs/tables", help="write tables here")
    args = ap.parse_args()

    ensure_dir(args.out)

    metric_files = find_metrics_json(args.root)
    if not metric_files:
        print("[WARN] No metrics.json found under:", args.root)
        return

    rows: List[Dict[str, Any]] = []
    for p in metric_files:
        j = load_json(p)
        j["_path"] = p
        # 兼容没有 method 字段的情况
        if "method" not in j:
            # 通过路径猜测 method
            # e.g., outputs/baselines/Threshold/seed42/metrics.json
            parts = p.replace("\\", "/").split("/")
            if "baselines" in parts:
                idx = parts.index("baselines")
                if idx + 1 < len(parts):
                    j["method"] = parts[idx + 1]
            else:
                j["method"] = "Unknown"
        rows.append(j)

    df = pd.DataFrame(rows)

    # 输出全量明细（每条 seed 一行）
    all_csv = os.path.join(args.out, "metrics_all_runs.csv")
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    # ===== Table 8：主结果（PPO vs Baselines）=====
    # 只要 method 字段存在，就能聚合
    table8 = to_mean_std(df, group_key="method")
    table8_csv = os.path.join(args.out, "Table8_main_results_mean_std.csv")
    table8.to_csv(table8_csv, index=False, encoding="utf-8-sig")
    print("[OK] Saved:")
    print(" -", all_csv)
    print(" -", table8_csv)
    if "case" in df.columns:
        table10 = to_mean_std(df, group_key="case")
        table10_csv = os.path.join(args.out, "Table10_ablation_mean_std.csv")
        table10.to_csv(table10_csv, index=False, encoding="utf-8-sig")
        print(" -", table10_csv)

        # ===== Table 9：Robustness under noise_test =====
    if "scenario" in df.columns:
        df_noise = df[df["scenario"] == "noise_test"].copy()
        if len(df_noise) > 0:
            table9 = to_mean_std(df_noise, group_key="method")
            table9_csv = os.path.join(args.out, "Table9_robust_noise_mean_std.csv")
            table9.to_csv(table9_csv, index=False, encoding="utf-8-sig")
            print(" -", table9_csv)




if __name__ == "__main__":
    main()
