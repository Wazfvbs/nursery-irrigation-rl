from __future__ import annotations

"""
make_tables.py (ref-aware)

适配你的目录结构：
- outputs/ablation_ref/...        -> Table10（消融），同时 ablation_ref/Full 也用于 Table8 的 PPO 主结果
- outputs/baselines_ref/...       -> Table8（基线）
- outputs/noise_test_ref/...      -> Table9（鲁棒/噪声）
- outputs/tables_ref/             -> 输出表格目录（你可以 --out 指定）

核心规则：
- Table8：nominal（scenario 为空）且 (case 为空 或 case == 'Full')，其中 case=='Full' 会被当作 method='PPO'
- Table10：所有 case 非空（包含 Full、w/o_*）
- Table9：scenario == 'noise_test'
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


CORE_COLS = [
    "TIR_ref",
    "IAE_mid_ref",
    "TotalIrrigation_mm",
    "StressDays_ref",
    "ActionTV",
]

SUPP_COLS = [
    "MAE_ref_mm",
    "RMSE_ref_mm",
    "UnderDays_ref",
    "ActionStd",
    "RewardMean",
]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_metrics_json(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "metrics.json":
                paths.append(os.path.join(dirpath, fn))
    return paths


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _startswith_any(s: str, prefixes: List[str]) -> bool:
    return any(s.startswith(p) for p in prefixes)


def infer_case_from_path(p: str) -> Optional[str]:
    parts = p.replace("\\", "/").split("/")
    # e.g. outputs/ablation_ref/Full/seed42/metrics.json
    for i, part in enumerate(parts):
        if _startswith_any(part, ["ablation", "ablation_ref"]):
            if i + 1 < len(parts):
                cand = parts[i + 1]
                # seed42/metrics.json 这种就不算 case
                if not cand.lower().startswith("seed") and cand not in ("metrics.json",):
                    return cand
    return None


def infer_scenario_from_path(p: str) -> Optional[str]:
    parts = p.replace("\\", "/").split("/")
    for part in parts:
        if _startswith_any(part, ["noise_test", "noise_test_ref"]):
            return "noise_test"
    return None  # nominal


def infer_method_from_path(p: str, inferred_case: Optional[str]) -> str:
    parts = p.replace("\\", "/").split("/")

    # baselines_ref
    for i, part in enumerate(parts):
        if _startswith_any(part, ["baselines", "baselines_ref"]):
            return parts[i + 1] if i + 1 < len(parts) else "Baselines"

    # noise_test_ref: 方法名缺失时，默认把它当 PPO（更符合 Table9 设计）
    for part in parts:
        if _startswith_any(part, ["noise_test", "noise_test_ref"]):
            return "PPO"

    # ablation_ref: Full -> PPO；其它 case -> Ablation::<case>
    for part in parts:
        if _startswith_any(part, ["ablation", "ablation_ref"]):
            if inferred_case == "Full":
                return "PPO"
            if inferred_case:
                return f"Ablation::{inferred_case}"
            return "Ablation"

    # ppo_runs（你现在没有，但保留兼容）
    for part in parts:
        if _startswith_any(part, ["ppo_runs", "ppo_runs_ref"]):
            return "PPO"

    return "Unknown"


def to_mean_std(df: pd.DataFrame, group_key: str, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame({group_key: sorted(df[group_key].dropna().unique())})

    # 强制数值化，避免 json 写成字符串导致聚合 nan
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    g = df.groupby(group_key)[cols].agg(["mean", "std"]).reset_index()
    g.columns = [f"{a}_{b}" if b else a for a, b in g.columns]

    out = pd.DataFrame()
    out[group_key] = g[group_key]

    for col in cols:
        m = g[f"{col}_mean"]
        s = g[f"{col}_std"]
        out[col] = m.map(lambda x: f"{x:.6f}" if pd.notna(x) else "nan") + " ± " + s.map(
            lambda x: f"{x:.6f}" if pd.notna(x) else "nan"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs", help="scan outputs/ for metrics.json")
    ap.add_argument("--out", type=str, default="outputs/tables_ref", help="write tables here")
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

        # scenario
        if "scenario" not in j or j.get("scenario") is None:
            j["scenario"] = infer_scenario_from_path(p)

        # case（若 json 没写，则从路径推断）
        if "case" not in j or not j.get("case"):
            inferred_case = infer_case_from_path(p)
            if inferred_case:
                j["case"] = inferred_case
        else:
            inferred_case = str(j.get("case"))

        # method
        if "method" not in j or not j.get("method"):
            j["method"] = infer_method_from_path(p, inferred_case if inferred_case else None)

        # 关键：让 ablation_ref/Full 作为 PPO 主结果进入 Table8
        if str(j.get("case", "")) == "Full":
            j["method"] = "PPO"

        rows.append(j)

    df = pd.DataFrame(rows)

    # 保存明细
    df.to_csv(os.path.join(args.out, "metrics_all_runs.csv"), index=False, encoding="utf-8-sig")

    # -------------------------
    # Table 8: 主结果（nominal + PPO(Full) + baselines）
    # -------------------------
    df_nominal = df.copy()
    df_nominal = df_nominal[df_nominal["scenario"].isna()].copy()

    # 排除消融的非 Full case（但保留 Full 作为 PPO）
    if "case" in df_nominal.columns:
        df_nominal = df_nominal[(df_nominal["case"].isna()) | (df_nominal["case"] == "Full")].copy()

    # 再排除任何 Ablation::<case>（以防 case 推断失败）
    df_nominal = df_nominal[~df_nominal["method"].astype(str).str.startswith("Ablation::")].copy()

    table8 = to_mean_std(df_nominal, "method", CORE_COLS)
    table8.to_csv(os.path.join(args.out, "Table8_main_results_mean_std.csv"), index=False, encoding="utf-8-sig")

    table8_supp = to_mean_std(df_nominal, "method", SUPP_COLS)
    table8_supp.to_csv(os.path.join(args.out, "Table8_main_results_supp_mean_std.csv"), index=False, encoding="utf-8-sig")

    # -------------------------
    # Table 10: 消融（ablation cases）
    # -------------------------
    if "case" in df.columns:
        df_ab = df[df["case"].notna()].copy()
        table10 = to_mean_std(df_ab, "case", CORE_COLS)
        table10.to_csv(os.path.join(args.out, "Table10_ablation_mean_std.csv"), index=False, encoding="utf-8-sig")

        table10_supp = to_mean_std(df_ab, "case", SUPP_COLS)
        table10_supp.to_csv(os.path.join(args.out, "Table10_ablation_supp_mean_std.csv"), index=False, encoding="utf-8-sig")

    # -------------------------
    # Table 9: 鲁棒性（noise_test）
    # -------------------------
    df_noise = df[df["scenario"] == "noise_test"].copy()
    if len(df_noise) > 0:
        table9 = to_mean_std(df_noise, "method", CORE_COLS)
        table9.to_csv(os.path.join(args.out, "Table9_robust_noise_mean_std.csv"), index=False, encoding="utf-8-sig")

        table9_supp = to_mean_std(df_noise, "method", SUPP_COLS)
        table9_supp.to_csv(os.path.join(args.out, "Table9_robust_noise_supp_mean_std.csv"), index=False, encoding="utf-8-sig")

    print("[OK] Tables written to:", args.out)


if __name__ == "__main__":
    main()
