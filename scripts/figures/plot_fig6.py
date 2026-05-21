# -*- coding: utf-8 -*-
"""
Fig.6 as three separate figures:
- fig6_a_dr_distribution.(png/pdf)
- fig6_b_irrigation_actions.(png/pdf)
- fig6_c_cumulative_irrigation.(png/pdf)

Expected input CSV columns (minimum):
- day: int in [1..H]
- Dr: root-zone depletion (mm)
- I: irrigation amount (mm/day)
- Dr_lo, Dr_hi: dynamic target interval bounds (mm)
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _read_csv(fp: str) -> pd.DataFrame:
    return pd.read_csv(fp)


def _collect_runs(pattern: str) -> dict:
    """Return {seed:int -> df}"""
    out = {}
    for fp in sorted(glob.glob(pattern)):
        seed = None
        for part in Path(fp).parts:
            if part.startswith("seed"):
                try:
                    seed = int(part.replace("seed", ""))
                except Exception:
                    seed = None
        if seed is None:
            seed = len(out)
        out[seed] = _read_csv(fp)
    if not out:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return out


def _pick_first_nonempty_pattern(patterns: list[str]) -> str:
    for p in patterns:
        if len(glob.glob(p)) > 0:
            return p
    # return first candidate for clearer downstream error
    return patterns[0]


def _stack_by_day(runs: dict, col: str) -> pd.DataFrame:
    rows = []
    for seed, df in runs.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in seed={seed}. Available: {list(df.columns)}")
        tmp = df[["day", col]].copy()
        tmp["seed"] = seed
        tmp = tmp.rename(columns={col: "value"})
        rows.append(tmp)
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["day"] = out["day"].astype(int)
    return out


def _mean_std_by_day(runs: dict, col: str) -> pd.DataFrame:
    stacked = _stack_by_day(runs, col)
    g = stacked.groupby("day")["value"]
    stat = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1)}).reset_index()
    stat["std"] = stat["std"].fillna(0.0)
    return stat.sort_values("day")


def _cumsum_mean_std(runs: dict, col: str) -> pd.DataFrame:
    per_seed = []
    for seed, df in runs.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in seed={seed}.")
        tmp = df[["day", col]].copy()
        tmp["day"] = tmp["day"].astype(int)
        tmp = tmp.sort_values("day")
        tmp["seed"] = seed
        tmp["cumsum"] = tmp[col].cumsum()
        per_seed.append(tmp[["day", "seed", "cumsum"]])
    per_seed = pd.concat(per_seed, ignore_index=True)

    g = per_seed.groupby("day")["cumsum"]
    stat = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1)}).reset_index()
    stat["std"] = stat["std"].fillna(0.0)
    return stat.sort_values("day")


def _set_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, out_png: str, out_pdf: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_pdf}")


def get_target_band_from_ppo(ppo_runs: dict):
    # Use median over ALL PPO steps/seeds (matches your current Fig6 logic)
    if "Dr_lo" not in next(iter(ppo_runs.values())).columns or "Dr_hi" not in next(iter(ppo_runs.values())).columns:
        raise KeyError("Need columns 'Dr_lo' and 'Dr_hi' in PPO trajectories to compute target band.")
    band_lo = float(np.median(pd.concat([df["Dr_lo"] for df in ppo_runs.values()], ignore_index=True)))
    band_hi = float(np.median(pd.concat([df["Dr_hi"] for df in ppo_runs.values()], ignore_index=True)))
    return band_lo, band_hi


def plot_fig6a_dr_distribution(ppo_runs, tuned_runs, vanilla_runs, out_prefix: str):
    _set_style()
    band_lo, band_hi = get_target_band_from_ppo(ppo_runs)

    dr_ppo = pd.concat([df["Dr"] for df in ppo_runs.values()], ignore_index=True).dropna().values
    dr_tuned = pd.concat([df["Dr"] for df in tuned_runs.values()], ignore_index=True).dropna().values
    dr_vanilla = pd.concat([df["Dr"] for df in vanilla_runs.values()], ignore_index=True).dropna().values

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axhspan(band_lo, band_hi, alpha=0.18)

    data = [dr_ppo, dr_tuned, dr_vanilla]
    labels = ["UC-PPO", "Tuned FAO-rule", "Fixed-target PPO"]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="^", markersize=9, markerfacecolor="tab:green", markeredgecolor="tab:green"),
        medianprops=dict(color="tab:orange", linewidth=2),
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        flierprops=dict(marker="o", markersize=6, markerfacecolor="none", markeredgecolor="black", alpha=0.9),
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.08)

    ax.set_ylabel(r"Root-zone depletion $D_{r,t}$ (mm)")
    ax.set_xlabel("Method")
    #ax.set_title(r"(a) $D_{r,t}$ distribution (nominal)", pad=8)

    ymin = min(0, np.min([dr_ppo.min(), dr_tuned.min(), dr_vanilla.min()]) - 1)
    ax.set_ylim(bottom=ymin)

    # clearer legend
    legend_handles = [
        Patch(facecolor="tab:blue", alpha=0.18, edgecolor="none",
              label=f"Target band [{band_lo:.1f}, {band_hi:.1f}] mm"),
        Line2D([0], [0], color="tab:orange", lw=2, label="Median"),
        Line2D([0], [0], marker="^", color="tab:green", lw=0, markersize=9, label="Mean"),
        Line2D([0], [0], marker="o", color="black", lw=0, markerfacecolor="none", label="Outlier"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout()
    _save(fig, out_prefix + "_a.png", out_prefix + "_a.pdf")


def plot_fig6b_irrigation_actions(
        ppo_runs, tuned_runs, vanilla_runs,
        out_prefix: str,
        irr_clip: float = 2.5
):
    _set_style()
    it_ppo = _mean_std_by_day(ppo_runs, "I")
    it_tuned = _mean_std_by_day(tuned_runs, "I")
    it_vanilla = _mean_std_by_day(vanilla_runs, "I")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    # --- fixed, consistent colors (manual but journal-friendly) ---
    c_ppo = "#1f77b4"   # blue
    c_tuned = "#2ca02c"   # green
    c_vanilla = "#ff7f0e"   # orange
    c_peak = "#d62728"  # red

    # ===================== Main axis: PPO clipped view =====================
    d = it_ppo["day"].to_numpy()
    m = it_ppo["mean"].to_numpy()
    s = it_ppo["std"].to_numpy()

    m_clip = np.minimum(m, irr_clip)

    l_ppo, = ax.plot(d, m_clip, color=c_ppo, linewidth=2.2, label="UC-PPO (mean, clipped)")
    ax.fill_between(
        d,
        np.maximum(0, m_clip - s),
        m_clip + s,
        color=c_ppo, alpha=0.18,
        label="PPO ± std"
    )

    clipped_mask = (m > irr_clip)
    if np.any(clipped_mask):
        # one legend entry only (no per-point text)
        ax.scatter(
            d[clipped_mask],
            np.full(np.sum(clipped_mask), irr_clip),
            s=32, color=c_peak, zorder=6,
            label=f"Clipped peaks"
        )

    ax.set_ylabel(r"Irrigation $I_t$ (mm/day)")
    ax.set_xlabel("Day")
    ax.set_ylim(0, irr_clip * 1.25)
    ax.grid(alpha=0.25)

    # 子图编号保留，但不使用标题（标题交给 subcaption）
    #ax.text(0.02, 0.96, "(b)", transform=ax.transAxes,ha="left", va="top", fontsize=12)

    # ===================== Inset: baselines full scale =====================
    axins = inset_axes(ax, width="38%", height="55%", loc="upper right", borderpad=1.0)

    # Tuned FAO-rule
    dd = it_tuned["day"].to_numpy()
    mm = it_tuned["mean"].to_numpy()
    ss = it_tuned["std"].to_numpy()
    l_tuned, = axins.plot(dd, mm, color=c_tuned, linewidth=1.8, label="Tuned FAO-rule")
    axins.fill_between(dd, np.maximum(0, mm - ss), mm + ss, color=c_tuned, alpha=0.12)

    # Fixed-target PPO
    dd = it_vanilla["day"].to_numpy()
    mm = it_vanilla["mean"].to_numpy()
    ss = it_vanilla["std"].to_numpy()
    l_vanilla, = axins.plot(dd, mm, color=c_vanilla, linewidth=1.8, label="Fixed-target PPO")
    axins.fill_between(dd, np.maximum(0, mm - ss), mm + ss, color=c_vanilla, alpha=0.12)

    # 去掉 inset 标题（你想删掉的小图标题）
    # axins.set_title("Baselines (full scale)", fontsize=10, pad=-6)

    axins.set_xlim(d.min(), d.max())
    ymax = max(
        float((it_tuned["mean"] + it_tuned["std"]).max()),
        float((it_vanilla["mean"] + it_vanilla["std"]).max()),
        1.0,
    )
    axins.set_ylim(0, ymax)
    axins.grid(alpha=0.20)
    axins.tick_params(labelsize=9)

    # ===================== One unified legend (main + baselines) =====================
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axins.get_legend_handles_labels()

    # 合并：主图图例 + inset 的两条 baseline 线
    ax.legend(h1 + h2, l1 + l2,loc="upper left", frameon=False, bbox_to_anchor=(0.08, 1.0))
    fig.tight_layout()
    _save(fig, out_prefix + "_b.png", out_prefix + "_b.pdf")


def plot_fig6c_cumulative_irrigation(ppo_runs, tuned_runs, vanilla_runs, out_prefix: str):
    _set_style()
    cum_ppo = _cumsum_mean_std(ppo_runs, "I")
    cum_tuned = _cumsum_mean_std(tuned_runs, "I")
    cum_vanilla = _cumsum_mean_std(vanilla_runs, "I")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    def plot_cum(stat: pd.DataFrame, label: str):
        dd = stat["day"].to_numpy()
        mm = stat["mean"].to_numpy()
        ss = stat["std"].to_numpy()
        ax.plot(dd, mm, linewidth=2, label=label)
        ax.fill_between(dd, np.maximum(0, mm - ss), mm + ss, alpha=0.18)

    plot_cum(cum_ppo, "UC-PPO")
    plot_cum(cum_tuned, "Tuned FAO-rule")
    plot_cum(cum_vanilla, "Fixed-target PPO")

    ax.set_ylabel(r"Cumulative irrigation $\sum_{t \leq d} I_t$ (mm)")
    ax.set_xlabel("Day")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    #ax.set_title(r"(c) Cumulative irrigation curve (mean ± std over seeds)", pad=8)

    fig.tight_layout()
    _save(fig, out_prefix + "_c.png", out_prefix + "_c.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Project root (where output/ exists)")
    ap.add_argument("--out", type=str, default="output/figures/fig6_nominal_failure",
                    help="Output path prefix, e.g., output/figures/fig6_nominal_failure")
    ap.add_argument("--clip", type=float, default=2.5, help="PPO clip value for panel (b)")
    args = ap.parse_args()

    root = Path(args.root)

    ppo_pat = _pick_first_nonempty_pattern([
        str(root / "output" / "ppo_runs" / "seed*" / "eval" / "trajectory.csv"),
        str(root / "output" / "robust_test_ref" / "PPO" / "nominal" / "seed*" / "trajectory.csv"),
        str(root / "output" / "ablation_ref" / "Full" / "seed*" / "eval" / "trajectory.csv"),
    ])
    tuned_pat = _pick_first_nonempty_pattern([
        str(root / "output" / "baselines" / "TunedFAORule" / "seed*" / "trajectory.csv"),
        str(root / "output" / "baselines_ref" / "TunedFAORule" / "seed*" / "trajectory.csv"),
    ])
    vanilla_pat = _pick_first_nonempty_pattern([
        str(root / "output" / "vanilla_ppo_runs" / "seed*" / "eval" / "trajectory.csv"),
        str(root / "output" / "robust_test_ref" / "FixedTargetPPO" / "nominal" / "seed*" / "trajectory.csv"),
        str(root / "output" / "robust_test_ref" / "VanillaPPO" / "nominal" / "seed*" / "trajectory.csv"),
    ])

    ppo_runs = _collect_runs(ppo_pat)
    tuned_runs = _collect_runs(tuned_pat)
    vanilla_runs = _collect_runs(vanilla_pat)

    out_prefix = str(Path(args.out))

    plot_fig6a_dr_distribution(ppo_runs, tuned_runs, vanilla_runs, out_prefix=out_prefix)
    plot_fig6b_irrigation_actions(ppo_runs, tuned_runs, vanilla_runs, out_prefix=out_prefix, irr_clip=args.clip)
    plot_fig6c_cumulative_irrigation(ppo_runs, tuned_runs, vanilla_runs, out_prefix=out_prefix)


if __name__ == "__main__":
    main()
