import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_SPECS = [
    ("PPO", "PPO-Optimized", "tab:blue", ["PPO", "PPO-Optimized"]),
    (
        "TunedFAORule",
        "Tuned FAO-rule",
        "tab:green",
        [
            "TunedFAORule",
            "Tuned FAO-rule",
            "Tuned FAO-rule (grid)",
            "Tuned FAO Rule",
            "Tuned FAO Rule (grid)",
        ],
    ),
    ("VanillaPPO", "Vanilla PPO", "tab:orange", ["VanillaPPO", "Vanilla PPO"]),
]

PANEL_SPECS = [
    {
        "tag": "(a)",
        "metric": "TIR_ref",
        "title": "Tracking stability under noise",
        "xlabel": "TIR_ref (%)",
        "scale": 100.0,
        "decimals": 2,
        "limit_fn": "tir",
    },
    {
        "tag": "(b)",
        "metric": "TotalIrrigation_mm",
        "title": "Irrigation cost under noise",
        "xlabel": "Total irrigation (mm)",
        "scale": 1.0,
        "decimals": 1,
        "limit_fn": "irrigation",
    },
    {
        "tag": "(c)",
        "metric": "StressDays_ref",
        "title": "Stress occurrence under noise",
        "xlabel": "StressDays_ref (#)",
        "scale": 1.0,
        "decimals": 2,
        "limit_fn": "stress_noise",
    },
]


def _norm_method_name(value):
    return re.sub(r"[\s_\-()]", "", str(value).lower())


def _pick_method_row(df, aliases):
    alias_set = {_norm_method_name(alias) for alias in aliases}
    for _, row in df.iterrows():
        if _norm_method_name(row["method"]) in alias_set:
            return row.copy()
    raise KeyError(f"Missing method alias from {aliases}. Available methods: {df['method'].tolist()}")


def parse_mean_std(value):
    if pd.isna(value):
        return np.nan, np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value), 0.0
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    if len(tokens) >= 2:
        return float(tokens[0]), float(tokens[1])
    if len(tokens) == 1:
        return float(tokens[0]), 0.0
    return np.nan, np.nan


def format_mean_std(mean, std, decimals):
    return f"{mean:.{decimals}f}+/-{std:.{decimals}f}"


def padded_limits(means, stds, margin=0.08, floor_zero=False, snap=None, min_span=None):
    lows = np.asarray(means) - np.asarray(stds)
    highs = np.asarray(means) + np.asarray(stds)
    lo = float(np.nanmin(lows))
    hi = float(np.nanmax(highs))
    span = hi - lo

    if not np.isfinite(span) or span <= 0.0:
        span = max(abs(hi), 1.0) * 0.1

    if min_span is not None and span < min_span:
        extra = 0.5 * (min_span - span)
        lo -= extra
        hi += extra
        span = hi - lo

    pad = max(span * margin, 0.02 * max(abs(lo), abs(hi), 1.0))
    lo -= pad
    hi += pad

    if floor_zero:
        lo = 0.0

    if snap:
        lo = math.floor(lo / snap) * snap
        hi = math.ceil(hi / snap) * snap

    if hi <= lo:
        hi = lo + max(min_span or 1.0, 1.0)

    return lo, hi


def tir_limits(means, stds):
    lows = np.asarray(means) - np.asarray(stds)
    highs = np.asarray(means) + np.asarray(stds)
    lo = max(98.5, float(np.nanmin(lows)) - 0.08)
    hi = min(100.2, float(np.nanmax(highs)) + 0.08)
    lo = math.floor(lo * 2.0) / 2.0
    hi = math.ceil(hi * 20.0) / 20.0
    if hi - lo < 0.5:
        lo = max(98.5, hi - 0.5)
    return lo, hi


def noise_stress_limits(means, stds):
    upper = float(np.nanmax(np.asarray(means) + np.asarray(stds)))
    if upper <= 2.0:
        return 0.0, 2.0
    return 0.0, max(3.0, math.ceil((upper * 1.1) / 0.5) * 0.5)


def get_limits(limit_fn, means, stds):
    if limit_fn == "tir":
        return tir_limits(means, stds)
    if limit_fn == "irrigation":
        return padded_limits(means, stds, margin=0.08, snap=2.0, min_span=8.0)
    if limit_fn == "stress_noise":
        return noise_stress_limits(means, stds)
    raise ValueError(f"Unknown limit function: {limit_fn}")


def extract_metric(rows, metric, scale):
    means = []
    stds = []
    for value in rows[metric]:
        mean, std = parse_mean_std(value)
        means.append(mean * scale)
        stds.append(std * scale)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def plot_panel(ax, labels, means, stds, colors, title, xlabel, xlim, decimals, show_labels):
    y = np.arange(len(labels), dtype=float)[::-1]
    span = xlim[1] - xlim[0]
    text_offset = 0.03 * span

    for ypos, mean, std, color in zip(y, means, stds, colors):
        ax.errorbar(
            mean,
            ypos,
            xerr=std,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=4,
            markersize=8.5,
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=3,
        )

        text_x = mean + text_offset
        halign = "left"
        if text_x > xlim[1] - 0.02 * span:
            text_x = mean - text_offset
            halign = "right"

        ax.text(
            text_x,
            ypos,
            format_mean_std(mean, std, decimals),
            ha=halign,
            va="center",
            fontsize=8.1,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_yticks(y)
    if show_labels:
        ax.set_yticklabels(labels)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_title(f"{title}", loc="left", pad=8)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table9", type=str, default="output/tables_ref/Table9_robust_noise_only_mean_std.csv")
    parser.add_argument("--out_prefix", type=str, default="output/figures/fig8_robustness")
    parser.add_argument("--note", type=str, default="Points and error bars show mean +/- std.")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    csv_path = Path(args.table9)
    df = pd.read_csv(csv_path)

    selected_rows = []
    display_labels = []
    colors = []
    for _, display_label, color, aliases in METHOD_SPECS:
        selected_rows.append(_pick_method_row(df, aliases))
        display_labels.append(display_label)
        colors.append(color)

    rows = pd.DataFrame(selected_rows).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), sharey=True)
    panel_ranges = {}

    for index, (ax, panel) in enumerate(zip(axes, PANEL_SPECS)):
        means, stds = extract_metric(rows, panel["metric"], panel["scale"])
        xlim = get_limits(panel["limit_fn"], means, stds)
        panel_ranges[panel["metric"]] = xlim
        plot_panel(
            ax,
            display_labels,
            means,
            stds,
            colors,
            f"{panel['tag']} {panel['title']}",
            panel["xlabel"],
            xlim,
            panel["decimals"],
            show_labels=index == 0,
        )

    axes[0].set_ylabel("Method")
    fig.text(0.99, 0.03, args.note, ha="right", fontsize=8.5, alpha=0.8)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[FIG8] CSV: {csv_path.resolve()}")
    print(f"[FIG8] Method order: {' -> '.join(display_labels)}")
    for panel in PANEL_SPECS:
        lo, hi = panel_ranges[panel["metric"]]
        print(f"[FIG8] {panel['metric']}: {lo:.2f} to {hi:.2f}")
    print(f"[OK] Saved: {out_png.resolve()}")
    print(f"[OK] Saved: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
