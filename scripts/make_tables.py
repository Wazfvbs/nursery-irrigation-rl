from __future__ import annotations

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
    "clip_rate",
    "ActionStd",
    "RewardMean",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_metrics_json(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn == "metrics.json":
                out.append(os.path.join(dirpath, fn))
    return out


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_parts(path: str) -> List[str]:
    return path.replace("\\", "/").split("/")


def infer_case_from_path(path: str) -> Optional[str]:
    parts = split_parts(path)
    for i, part in enumerate(parts):
        if part in ("ablation", "ablation_ref"):
            if i + 1 < len(parts):
                cand = parts[i + 1]
                if not cand.lower().startswith("seed") and cand != "metrics.json":
                    return cand
    return None


def infer_method_from_path(path: str, case_name: Optional[str]) -> str:
    parts = split_parts(path)

    for i, part in enumerate(parts):
        if part in ("baselines", "baselines_ref") and i + 1 < len(parts):
            return parts[i + 1]

    for i, part in enumerate(parts):
        if part == "robust_test_ref" and i + 1 < len(parts):
            return parts[i + 1]

    for part in parts:
        if part in ("vanilla_ppo_runs", "vanilla_ppo_runs_ref"):
            return "FixedTargetPPO"
        if part in ("ppo_runs", "ppo_runs_ref"):
            return "UC-PPO"

    for part in parts:
        if part in ("ablation", "ablation_ref"):
            if case_name == "Full":
                return "UC-PPO"
            if case_name:
                return f"Ablation::{case_name}"
            return "Ablation"

    return "Unknown"


def normalize_method_name(method: Any, path: str, case_name: Optional[str]) -> str:
    name = str(method)
    compact = name.replace(" ", "").replace("-", "").replace("_", "").lower()
    parts = split_parts(path)

    if compact in ("vanillappo", "fixedtargetppo"):
        return "FixedTargetPPO"
    if compact in ("ppo", "ppooptimized", "ucppo") and (
        "ppo_runs" in parts or "ppo_runs_ref" in parts or case_name == "Full"
    ):
        return "UC-PPO"
    return name


def infer_scenario_from_path(path: str) -> str:
    parts = split_parts(path)
    if "robust_test_ref" in parts:
        return "robustness"
    if "noise_test_ref" in parts or "noise_test" in parts:
        return "robustness"
    return "nominal"


def infer_setting_from_path(path: str) -> str:
    parts = split_parts(path)
    for i, part in enumerate(parts):
        if part == "robust_test_ref" and i + 2 < len(parts):
            return parts[i + 2]
    if "noise_test_ref" in parts or "noise_test" in parts:
        return "noise_only"
    return "nominal"


def normalize_scenario(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "nominal"
    ss = str(s).strip().lower()
    if ss in ("", "none", "null", "nan", "nominal"):
        return "nominal"
    if ss in ("noise_test", "robustness"):
        return "robustness"
    if ss in ("noise_only", "et0_only", "noise_et0"):
        return "robustness"
    return ss


def normalize_setting(s: Any, scenario: str) -> str:
    if scenario != "robustness":
        return "nominal"
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "noise_only"
    ss = str(s).strip().lower()
    alias = {
        "noise": "noise_only",
        "noise-only": "noise_only",
        "noiseonly": "noise_only",
        "obs_noise": "noise_only",
        "et0": "et0_only",
        "et0-only": "et0_only",
        "et0only": "et0_only",
        "noise+et0": "noise_et0",
        "noise-et0": "noise_et0",
    }
    return alias.get(ss, ss if ss else "noise_only")


def format_mean_std_value(mean: Any, std: Any, count: Any) -> str:
    if not pd.notna(mean):
        return "nan"
    if not pd.notna(std) or int(count) <= 1:
        return f"{float(mean):.6f}"
    return f"{float(mean):.6f} +/- {float(std):.6f}"


def to_mean_std(df: pd.DataFrame, group_key: str, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if group_key not in df.columns:
        return pd.DataFrame()
    if not cols:
        return pd.DataFrame({group_key: sorted(df[group_key].dropna().unique())})

    local = df.copy()
    for c in cols:
        local[c] = pd.to_numeric(local[c], errors="coerce")

    g = local.groupby(group_key)[cols].agg(["mean", "std", "count"]).reset_index()
    g.columns = [f"{a}_{b}" if b else a for a, b in g.columns]

    out = pd.DataFrame()
    out[group_key] = g[group_key]
    for col in cols:
        out[col] = [
            format_mean_std_value(m, s, n)
            for m, s, n in zip(g[f"{col}_mean"], g[f"{col}_std"], g[f"{col}_count"])
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs")
    ap.add_argument("--out", type=str, default="outputs/tables_ref")
    ap.add_argument(
        "--fix_metrics_metadata",
        action="store_true",
        help="Rewrite metrics.json files with normalized method/scenario/setting metadata.",
    )
    args = ap.parse_args()

    ensure_dir(args.out)

    metric_files = find_metrics_json(args.root)
    if not metric_files:
        print(f"[WARN] no metrics.json found under: {args.root}")
        return

    rows: List[Dict[str, Any]] = []
    for path in metric_files:
        j = load_json(path)
        j["_path"] = path

        case_name = j.get("case") or infer_case_from_path(path)
        if case_name:
            j["case"] = case_name

        if not j.get("method"):
            j["method"] = infer_method_from_path(path, case_name)
        if str(j.get("case", "")) == "Full":
            j["method"] = "UC-PPO"
        j["method"] = normalize_method_name(j["method"], path, case_name)

        raw_scenario = j.get("scenario")
        scenario = normalize_scenario(raw_scenario)
        if "scenario" not in j:
            scenario = infer_scenario_from_path(path)
        j["scenario"] = scenario

        setting = j.get("setting")
        if not setting:
            if isinstance(raw_scenario, str) and raw_scenario.strip().lower() in ("noise_only", "et0_only", "noise_et0"):
                setting = raw_scenario
            else:
                setting = infer_setting_from_path(path)
        j["setting"] = normalize_setting(setting, scenario=scenario)

        if args.fix_metrics_metadata:
            persisted = load_json(path)
            changed = False
            keys = ("case", "scenario", "setting") if case_name else ("method", "scenario", "setting")
            for key in keys:
                if key in j and persisted.get(key) != j[key]:
                    persisted[key] = j[key]
                    changed = True
            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(persisted, f, indent=2, ensure_ascii=False)

        rows.append(j)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "metrics_all_runs.csv"), index=False, encoding="utf-8-sig")

    # Table 8: nominal only. Prefer the official main run; use ablation Full only as a fallback.
    df_nominal = df[df["scenario"] == "nominal"].copy()
    if "case" in df_nominal.columns:
        has_main_ucppo = (
            (df_nominal["method"].astype(str) == "UC-PPO")
            & (df_nominal["case"].isna())
        ).any()
        if has_main_ucppo:
            df_nominal = df_nominal[df_nominal["case"].isna()].copy()
        else:
            df_nominal = df_nominal[(df_nominal["case"].isna()) | (df_nominal["case"] == "Full")].copy()
    df_nominal = df_nominal[~df_nominal["method"].astype(str).str.startswith("Ablation::")].copy()

    table8 = to_mean_std(df_nominal, "method", CORE_COLS)
    table8.to_csv(os.path.join(args.out, "Table8_main_results_mean_std.csv"), index=False, encoding="utf-8-sig")
    table8_supp = to_mean_std(df_nominal, "method", SUPP_COLS)
    table8_supp.to_csv(
        os.path.join(args.out, "Table8_main_results_supp_mean_std.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Table 10: ablation
    if "case" in df.columns:
        df_ab = df[df["case"].notna()].copy()
        if len(df_ab) > 0:
            table10 = to_mean_std(df_ab, "case", CORE_COLS)
            table10.to_csv(os.path.join(args.out, "Table10_ablation_mean_std.csv"), index=False, encoding="utf-8-sig")
            table10_supp = to_mean_std(df_ab, "case", SUPP_COLS)
            table10_supp.to_csv(
                os.path.join(args.out, "Table10_ablation_supp_mean_std.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    # Table 9: robustness by setting
    df_rob = df[df["scenario"] == "robustness"].copy()
    if len(df_rob) > 0:
        settings = sorted(df_rob["setting"].dropna().unique())

        combined = df_rob.copy()
        combined["method_setting"] = combined["method"].astype(str) + "@" + combined["setting"].astype(str)
        table9_all = to_mean_std(combined, "method_setting", CORE_COLS)
        table9_all.to_csv(
            os.path.join(args.out, "Table9_robust_by_setting_mean_std.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        table9_all_supp = to_mean_std(combined, "method_setting", SUPP_COLS)
        table9_all_supp.to_csv(
            os.path.join(args.out, "Table9_robust_by_setting_supp_mean_std.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        for setting in settings:
            df_s = df_rob[df_rob["setting"] == setting].copy()
            if len(df_s) == 0:
                continue
            t_main = to_mean_std(df_s, "method", CORE_COLS)
            t_supp = to_mean_std(df_s, "method", SUPP_COLS)

            main_name = f"Table9_robust_{setting}_mean_std.csv"
            supp_name = f"Table9_robust_{setting}_supp_mean_std.csv"
            t_main.to_csv(os.path.join(args.out, main_name), index=False, encoding="utf-8-sig")
            t_supp.to_csv(os.path.join(args.out, supp_name), index=False, encoding="utf-8-sig")

            # Backward-compatible alias for existing plotting scripts.
            if setting == "noise_only":
                t_main.to_csv(
                    os.path.join(args.out, "Table9_robust_noise_mean_std.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                t_supp.to_csv(
                    os.path.join(args.out, "Table9_robust_noise_supp_mean_std.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

    print(f"[OK] tables written to: {args.out}")


if __name__ == "__main__":
    main()
