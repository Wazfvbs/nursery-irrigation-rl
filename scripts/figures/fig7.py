# make_fig7_schemeA.py
# Fig.7 (Scheme A): 2×2 summary panels with axis clipping (to handle huge scale gaps).
# Input: Table8_main_results_mean_std.csv (cells formatted like "mean ± std")
#
# Example:
#   python make_fig7_schemeA.py --csv Table8_main_results_mean_std.csv --out fig7_main_results_schemeA_v1

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_mean_std(x):
    """
    Parse strings like:
      "99.9 ± 0.3"
      "0.000000 ± 0.000000"
    Return (mean, std) as floats.
    If std is missing, return (value, 0).
    """
    if isinstance(x, (int, float, np.number)):
        return float(x), 0.0
    s = str(x).strip()
    # normalize
    s = s.replace("+/-", "±").replace("+/-", "±")
    # capture two floats if present
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(m) >= 2 and "±" in s:
        return float(m[0]), float(m[1])
    if len(m) >= 1:
        return float(m[0]), 0.0
    return float("nan"), float("nan")


def fmt_label(mean, std, decimals=1, zero_std_hide=True):
    """Format value label. If std==0 and zero_std_hide=True -> show only mean."""
    if np.isnan(mean) or np.isnan(std):
        return "nan"
    if zero_std_hide and abs(std) < 1e-12:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def bar_panel(
        ax,
        means,
        stds,
        labels,
        colors,
        ylabel,
        panel_tag,
        ylim=None,
        clip_ymax=None,
        label_decimals=1,
        note_clip=False,
):
    x = np.arange(len(means))
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    clipped = np.zeros_like(means, dtype=bool)
    display = means.copy()

    # clip bars that exceed axis range (Scheme A)
    if clip_ymax is not None:
        clipped = means > clip_ymax
        # draw clipped bars slightly below the top so the triangle is visible
        display[clipped] = clip_ymax * 0.97

    # bars
    bars = ax.bar(
        x, display, width=0.65,
        color=colors, edgecolor="black", linewidth=0.6
    )

    # error bars for non-clipped bars
    mask = ~clipped
    if np.any(mask):
        ax.errorbar(
            x[mask], display[mask], yerr=stds[mask],
            fmt="none", ecolor="black", elinewidth=1.2,
            capsize=4, capthick=1.2, zorder=3
        )

    # triangle markers for clipped bars
    if clip_ymax is not None and np.any(clipped):
        ax.scatter(
            x[clipped],
            np.full(np.sum(clipped), clip_ymax * 0.985),
            marker="^",
            s=70,
            color="black",
            zorder=4
        )
        if note_clip:
            ax.text(
                0.98, 0.04,
                "▲ indicates value exceeds axis range",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9
            )

    # labels on bars
    for i, (b, m, s, is_clip) in enumerate(zip(bars, means, stds, clipped)):
        txt = fmt_label(m, s, decimals=label_decimals, zero_std_hide=True)

        if clip_ymax is not None and is_clip:
            # put text inside the clipped bar
            y = clip_ymax * 0.88
            ax.text(b.get_x() + b.get_width()/2, y, txt,
                    ha="center", va="center", fontsize=10)
        else:
            # normal: put text above the bar top (or error bar top)
            y = b.get_height()
            pad = (ylim[1] - ylim[0]) * 0.02 if ylim else max(1.0, y * 0.05)
            ax.text(b.get_x() + b.get_width()/2, y + pad, txt,
                    ha="center", va="bottom", fontsize=10)

    # axes styling
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # panel tag
    ax.text(0.02, 0.96, panel_tag, transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Table8_main_results_mean_std.csv")
    ap.add_argument("--out", default="fig7_main_results_schemeA_v1", help="output basename (no extension)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    # Expecting methods in the CSV like: PPO, FAORule, Threshold, Calendar
    # And columns like: TIR_ref, TotalIrrigation_mm, StressDays_ref, ActionTV_mm
    # Desired plotting order: PPO, FAO-rule, Threshold-rule, Fixed-interval (Calendar)
    method_order = ["PPO", "FAORule", "Threshold", "Calendar"]
    pretty_labels = ["PPO-Optimized", "FAO-rule", "Threshold-rule", "Fixed-interval\n(Calendar)"]
    # Colors: PPO (blue), FAO-rule (green), Threshold-rule (orange), Calendar (gray)
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:gray"]

    df = df.set_index("method").loc[method_order].reset_index()

    # Parse metrics
    tir_mean, tir_std = [], []
    tir_col = "TIR_ref"
    for v in df[tir_col]:
        m, s = parse_mean_std(v)
        tir_mean.append(m * 100.0)   # -> %
        tir_std.append(s * 100.0)

    irr_mean, irr_std = [], []
    for v in df["TotalIrrigation_mm"]:
        m, s = parse_mean_std(v)
        irr_mean.append(m)
        irr_std.append(s)

    stress_mean, stress_std = [], []
    for v in df["StressDays_ref"]:
        m, s = parse_mean_std(v)
        stress_mean.append(m)
        stress_std.append(s)

    tv_mean, tv_std = [], []
    for v in df["ActionTV"]:
        m, s = parse_mean_std(v)
        tv_mean.append(m)
        tv_std.append(s)

    # ===== Plot (Scheme A) =====
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=200)

    # (a) TIR_ref (%)
    bar_panel(
        axes[0, 0],
        tir_mean, tir_std,
        pretty_labels, colors,
        ylabel=r"Within-target ratio  $\mathrm{TIR}_{ref}$ (%)  (higher is better)",
        panel_tag="(a)",
        ylim=(0, 105),
        clip_ymax=None,
        label_decimals=1,
        note_clip=False,
    )

    # (b) Total irrigation (mm) — clipped
    bar_panel(
        axes[0, 1],
        irr_mean, irr_std,
        pretty_labels, colors,
        ylabel=r"Total irrigation (mm)  (lower is better)",
        panel_tag="(b)",
        ylim=(0, 200),
        clip_ymax=200,
        label_decimals=1,
        note_clip=True,
    )

    # (c) Stress days (#)
    bar_panel(
        axes[1, 0],
        stress_mean, stress_std,
        pretty_labels, colors,
        ylabel=r"Stress days$_{ref}$ (#)  (lower is better)",
        panel_tag="(c)",
        ylim=(0, 35),
        clip_ymax=None,
        label_decimals=1,
        note_clip=False,
    )

    # (d) ActionTV (mm) — clipped
    bar_panel(
        axes[1, 1],
        tv_mean, tv_std,
        pretty_labels, colors,
        ylabel=r"Action variability  ActionTV (mm)  (lower is better)",
        panel_tag="(d)",
        ylim=(0, 250),
        clip_ymax=250,
        label_decimals=1,
        note_clip=True,
    )

    fig.tight_layout()
    out_base = Path(args.out)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_base.with_suffix('.png')}")
    print(f"Saved: {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
