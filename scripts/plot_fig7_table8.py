import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------
def parse_mean_std(x):
    """
    Parse 'mean ± std' -> (mean, std)
    Also supports numeric input.
    """
    if isinstance(x, (int, float, np.floating)):
        return float(x), 0.0
    s = str(x).strip()
    parts = re.split(r"\s*±\s*", s)
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std


def set_sci_style():
    # clean, submission-friendly style
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 350,
        "font.family": "DejaVu Sans",  # safe fallback on Windows/Linux
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
    })


def get_col(df, candidates):
    """
    Find first existing column name from candidates.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Cannot find any of columns: {candidates}")


def pick_row(df, method_name):
    """
    Find row by method label. Supports small variants.
    """
    # normalize
    m = method_name.lower().replace(" ", "")
    for i in range(len(df)):
        s = str(df.loc[i, "method"]).lower().replace(" ", "")
        if s == m:
            return df.loc[i]
    # allow 'FAORule' vs 'FAO Rule'
    for i in range(len(df)):
        s = str(df.loc[i, "method"]).lower().replace(" ", "")
        if m in s or s in m:
            return df.loc[i]
    raise ValueError(f"Cannot find method '{method_name}' in df['method'].")


# -------------------------
# Plotting
# -------------------------
def plot_bar(ax, labels, means, stds, colors, edgecolors, ylabel, title,
             ylim=None, annotate=True, fmt="{:.2f}", unit=""):
    x = np.arange(len(labels))
    width = 0.60

    bars = ax.bar(
        x, means, width=width,
        yerr=stds, capsize=5,
        color=colors,
        edgecolor=edgecolors,
        linewidth=1.0,
        alpha=0.95,
        zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # annotate only means (clean!)
    if annotate:
        y_top = ax.get_ylim()[1]
        dy = 0.015 * (y_top - ax.get_ylim()[0])
        for xi, yi in zip(x, means):
            ax.text(xi, yi + dy, fmt.format(yi) + unit,
                    ha="center", va="bottom", fontsize=9)

    # spines thickness
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)


def add_info_box(ax, text, x=0.5, y=0.98, ha="center", va="top"):
    """
    Put info box INSIDE axes (top empty area), so it never overlaps the title.
    """
    bbox = dict(boxstyle="round,pad=0.35", fc="white", ec="#444444", alpha=0.92)
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha, va=va,
        bbox=bbox
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table8", type=str, required=True,
                    help="Path to Table8_main_results_mean_std.csv")
    ap.add_argument("--out_dir", type=str, default="figures",
                    help="Output directory")
    ap.add_argument("--name", type=str, default="fig7_main_results",
                    help="Output filename prefix (without extension)")
    args = ap.parse_args()

    set_sci_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.table8)

    # column names may vary slightly — be robust
    col_method = get_col(df, ["method", "Method", "algo"])
    if col_method != "method":
        df = df.rename(columns={col_method: "method"})

    col_irrig = get_col(df, ["TotalIrrigation_mm", "TotalIrrigation", "Irrigation_mm"])
    col_within = get_col(df, ["WithinTargetRatio", "WithinTarget", "TargetRatio"])

    # methods for Fig.7 (fixed order)
    labels = ["Threshold", "FAO Rule", "PPO"]

    # extract mean/std
    irrig_means, irrig_stds = [], []
    within_means, within_stds = [], []

    # mapping for names inside csv
    csv_map = {
        "Threshold": "Threshold",
        "FAO Rule": "FAORule",  # many of your files use FAORule
        "PPO": "PPO"
    }

    for lab in labels:
        row = pick_row(df, csv_map[lab])
        m, s = parse_mean_std(row[col_irrig])
        irrig_means.append(m)
        irrig_stds.append(s)

        m, s = parse_mean_std(row[col_within])
        within_means.append(m * 100.0)  # to %
        within_stds.append(s * 100.0)

    irrig_means = np.array(irrig_means)
    irrig_stds = np.array(irrig_stds)
    within_means = np.array(within_means)
    within_stds = np.array(within_stds)

    # derive key annotations
    ppo_irrig = irrig_means[2]
    thr_irrig = irrig_means[0]
    fao_irrig = irrig_means[1]
    save_thr = 100.0 * (thr_irrig - ppo_irrig) / thr_irrig
    save_fao = 100.0 * (fao_irrig - ppo_irrig) / fao_irrig

    ppo_within = within_means[2] / 100.0
    thr_within = within_means[0] / 100.0
    fao_within = within_means[1] / 100.0
    gain_thr = ppo_within / max(thr_within, 1e-12)
    gain_fao = ppo_within / max(fao_within, 1e-12)

    # colors: PPO highlight, baselines grey
    PPO_COLOR = "#1F77B4"
    BASE_COLOR = "#B3B3B3"
    EDGE_PPO = "#222222"
    EDGE_BASE = "#666666"

    colors = [BASE_COLOR, BASE_COLOR, PPO_COLOR]
    edges = [EDGE_BASE, EDGE_BASE, EDGE_PPO]

    # figure: 2 subplots vertical
    fig = plt.figure(figsize=(7.2, 6.2))
    fig.subplots_adjust(top=0.94)

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)

    # (a) irrigation
    ax1 = fig.add_subplot(gs[0, 0])
    plot_bar(
        ax1,
        labels=labels,
        means=irrig_means,
        stds=irrig_stds,
        colors=colors,
        edgecolors=edges,
        ylabel="Total irrigation (mm)",
        title="(a) Irrigation amount (lower is better)",
        ylim=(0, max(irrig_means + irrig_stds) * 1.15),
        annotate=True,
        fmt="{:.2f}",
        unit=""
    )
    add_info_box(
        ax1,
        f"PPO saves ~{save_fao:.1f}% vs FAO Rule\n~{save_thr:.1f}% vs Threshold",
        x=0.82, y=0.98, ha="center"
    )

    # (b) within-target ratio
    ax2 = fig.add_subplot(gs[1, 0])
    plot_bar(
        ax2,
        labels=labels,
        means=within_means,
        stds=within_stds,
        colors=colors,
        edgecolors=edges,
        ylabel="Within-target ratio (%)",
        title="(b) Within-target ratio (higher is better)",
        ylim=(0, 105),
        annotate=True,
        fmt="{:.2f}",
        unit="%"
    )
    add_info_box(
        ax2,
        f"PPO improves tracking by ~{gain_thr:.0f}× vs Threshold\nand ~{gain_fao:.0f}× vs FAO Rule",
        x=0.08, y=0.98, ha="left"
    )
    #ax1.set_title("(a) Irrigation amount (lower is better)", pad=16)
    #ax2.set_title("(b) Within-target ratio (higher is better)", pad=16)

# small footnote
    fig.text(0.985, 0.015, "mean ± std", ha="right", va="bottom", fontsize=9, alpha=0.7)

    # save
    out_png = out_dir / f"{args.name}.png"
    out_pdf = out_dir / f"{args.name}.pdf"
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print("[OK] Saved:", out_png)
    print("[OK] Saved:", out_pdf)


if __name__ == "__main__":
    main()
