from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42


def parse_mean_std(x) -> Tuple[float, float]:
    if pd.isna(x):
        return float("nan"), float("nan")
    s = str(x).strip()
    s = re.sub(r"\s*(卤|±)\s*", "+/-", s)
    parts = re.split(r"\s*\+/-\s*", s)
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) >= 1:
        return float(nums[0]), (float(nums[1]) if len(nums) >= 2 else 0.0)
    raise ValueError(f"cannot parse mean/std: {x}")


def build_mean_std(df: pd.DataFrame, col: str, case_col: str = "case") -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        case = str(row[case_col])
        out[case] = parse_mean_std(row[col])
    return out


def delta_vs_full(
    ms_dict: Dict[str, Tuple[float, float]],
    *,
    case_order: List[str],
    case_label: Dict[str, str],
    metric_name: str,
) -> Tuple[List[str], List[float], List[float]]:
    if "Full" not in ms_dict:
        raise KeyError(f"[{metric_name}] missing 'Full' row.")
    full_mean, full_std = ms_dict["Full"]

    xs, ys, es = [], [], []
    for c in case_order:
        if c not in ms_dict:
            raise KeyError(f"[{metric_name}] missing '{c}' row.")
        m, s = ms_dict[c]
        xs.append(case_label.get(c, c))
        ys.append(m - full_mean)
        es.append(math.sqrt(s * s + full_std * full_std))
    return xs, ys, es


def maybe_flip(metric_key: str, ys: List[float], es: List[float], use_improvement_sign: bool):
    if not use_improvement_sign:
        return ys, es
    negate = metric_key in {
        "TotalIrrigation_mm",
        "StressDays_ref",
        "UnderDays_ref",
        "ActionTV",
        "ActionStd",
        "MAE_ref_mm",
        "RMSE_ref_mm",
    }
    if negate:
        ys = [-v for v in ys]
    return ys, es


def plot_grid(
    df: pd.DataFrame,
    metrics: List[Tuple[str, str, bool]],
    *,
    case_order: List[str],
    case_label: Dict[str, str],
    use_improvement_sign: bool,
    out_png: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    for ax, (key, ylabel, to_pp) in zip(axes, metrics):
        ms = build_mean_std(df, key)
        xlab, ys, es = delta_vs_full(ms, case_order=case_order, case_label=case_label, metric_name=key)
        if to_pp:
            ys = [v * 100.0 for v in ys]
            es = [v * 100.0 for v in es]
        ys, es = maybe_flip(key, ys, es, use_improvement_sign=use_improvement_sign)

        ax.bar(xlab, ys, yerr=es, capsize=4)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_csv", type=str, default="outputs/tables_ref/Table10_ablation_mean_std.csv")
    ap.add_argument("--supp_csv", type=str, default="outputs/tables_ref/Table10_ablation_supp_mean_std.csv")
    ap.add_argument("--out_main", type=str, default="outputs/figures/Fig9_ablation_main.png")
    ap.add_argument("--out_supp", type=str, default="outputs/figures/Fig9_ablation_supp.png")
    ap.add_argument("--use_improvement_sign", action="store_true")
    args = ap.parse_args()

    main_df = pd.read_csv(args.main_csv)
    supp_df = pd.read_csv(args.supp_csv)

    case_order = ["wo_Shaping", "wo_Target", "wo_UCB"]
    case_label = {"wo_Shaping": "w/o shaping", "wo_Target": "w/o target", "wo_UCB": "w/o UCB"}

    metrics_main = [
        ("TIR_ref", r"$\Delta$ within-target (pp)", True),
        ("TotalIrrigation_mm", r"$\Delta$ total irrigation (mm)", False),
        ("StressDays_ref", r"$\Delta$ stress days (days)", False),
        ("ActionTV", r"$\Delta$ action TV", False),
    ]
    metrics_supp = [
        ("MAE_ref_mm", r"$\Delta$ MAE (mm)", False),
        ("RMSE_ref_mm", r"$\Delta$ RMSE (mm)", False),
        ("UnderDays_ref", r"$\Delta$ under-days (days)", False),
        ("ActionStd", r"$\Delta$ action std", False),
    ]

    plot_grid(
        main_df,
        metrics_main,
        case_order=case_order,
        case_label=case_label,
        use_improvement_sign=bool(args.use_improvement_sign),
        out_png=Path(args.out_main),
    )
    plot_grid(
        supp_df,
        metrics_supp,
        case_order=case_order,
        case_label=case_label,
        use_improvement_sign=bool(args.use_improvement_sign),
        out_png=Path(args.out_supp),
    )


if __name__ == "__main__":
    main()
