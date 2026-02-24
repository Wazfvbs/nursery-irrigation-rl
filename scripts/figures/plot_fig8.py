# make_fig8_noise_robustness_brokenaxis.py
# Reproduce Fig.8 (noise robustness) from Table 9 mean±std CSV.
# Requirements: pandas, numpy, matplotlib

import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_mean_std(x):
    """
    Parse strings like '0.963333 ± 0.054193' into (mean, std).
    Also supports '+/-' variants.
    """
    if pd.isna(x):
        return np.nan, np.nan
    s = str(x).strip()
    s = s.replace("±", "+/-")
    parts = re.split(r"\+/-", s)
    if len(parts) < 2:
        # If only mean is provided, treat std as 0
        return float(parts[0].strip()), 0.0
    mean = float(parts[0].strip())
    std = float(parts[1].strip())
    return mean, std


def add_break_marks(ax_left, ax_right, d=0.012, lw=1.0):
    """Draw diagonal break marks between two axes."""
    kwargs_left = dict(transform=ax_left.transAxes, color="k", clip_on=False, lw=lw)
    ax_left.plot((1 - d, 1 + d), (-d, +d), **kwargs_left)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_left)

    kwargs_right = dict(transform=ax_right.transAxes, color="k", clip_on=False, lw=lw)
    ax_right.plot((-d, +d), (-d, +d), **kwargs_right)
    ax_right.plot((-d, +d), (1 - d, 1 + d), **kwargs_right)


def style_broken_axes(ax_left, ax_right):
    """Hide touching spines and y labels on right axis."""
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(labelleft=False)
    ax_right.yaxis.set_ticks_position("none")
    ax_right.yaxis.set_ticklabels([])


def plot_points_with_errorbars(ax, xs, ys, xerr, colors=None, ms=7, cap=3):
    """Plot points with horizontal error bars. Supports a single color or a list of colors per point."""
    # colors can be a single color string or a list of colors per point
    if colors is None:
        colors = ["k"] * len(xs)
    # If a single color is provided, broadcast it
    if isinstance(colors, (str, tuple)):
        colors = [colors] * len(xs)
    for x, y, xe, c in zip(xs, ys, xerr, colors):
        ax.errorbar(
            x, y, xerr=xe,
            fmt="o", color=c, ecolor=c,
            elinewidth=1.0, capsize=cap, markersize=ms
        )


def center_row_title_and_xlabel(fig, ax_left, ax_right, title=None, xlabel=None,
                                title_offset=0.02, xlabel_offset=0.03, fontsize=10):
    """Place a centered title and/or xlabel across two adjacent axes.

    - fig: the Figure object
    - ax_left, ax_right: the left and right axes in the row (can be the same axis)
    - title, xlabel: strings to place (None to skip)
    - title_offset, xlabel_offset: vertical offsets in figure coords
    """
    # get positions in figure coordinates
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    x_center = 0.5 * (pos_l.x0 + pos_r.x1)

    if title is not None:
        y_top = max(pos_l.y1, pos_r.y1)
        fig.text(x_center, y_top + title_offset, title, ha="center", va="bottom", fontsize=fontsize)

    if xlabel is not None:
        y_bottom = min(pos_l.y0, pos_r.y0)
        fig.text(x_center, y_bottom - xlabel_offset, xlabel, ha="center", va="top", fontsize=fontsize)


def main(args):
    df = pd.read_csv(args.table9)

    # Required columns
    required = {"method", "TIR_ref", "TotalIrrigation_mm", "StressDays_ref"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing columns in {args.table9}: {missing}")

    # Standardize method labels & ordering (top to bottom)
    method_map = {
        "PPO": "PPO",
        "Threshold": "Threshold-rule",
        "FAORule": "FAO-rule",
    }
    df["method_disp"] = df["method"].map(method_map).fillna(df["method"])
    order = ["PPO", "Threshold-rule", "FAO-rule"]
    df["method_disp"] = pd.Categorical(df["method_disp"], categories=order, ordered=True)
    df = df.sort_values("method_disp")

    # y positions: PPO at top (2), Threshold (1), FAO (0)
    y_map = {m: (len(order) - 1 - i) for i, m in enumerate(order)}  # PPO->2, Threshold->1, FAO->0
    ys = np.array([y_map[m] for m in df["method_disp"].astype(str).tolist()], dtype=float)
    ytick_pos = [y_map[m] for m in order]
    ytick_lab = order

    # color mapping for methods: PPO blue, FAO-rule green, Threshold-rule orange
    color_map = {"PPO": "tab:blue", "FAO-rule": "tab:green", "Threshold-rule": "tab:orange"}
    colors = [color_map.get(m, "k") for m in df["method_disp"].astype(str).tolist()]

    # Parse metrics
    tir_mean, tir_std = zip(*df["TIR_ref"].map(parse_mean_std))
    tir_mean = np.array(tir_mean) * 100.0
    tir_std = np.array(tir_std) * 100.0

    irr_mean, irr_std = zip(*df["TotalIrrigation_mm"].map(parse_mean_std))
    irr_mean = np.array(irr_mean)
    irr_std = np.array(irr_std)

    sd_mean, sd_std = zip(*df["StressDays_ref"].map(parse_mean_std))
    sd_mean = np.array(sd_mean)
    sd_std = np.array(sd_std)

    # --- Figure layout (3 rows): (a) broken axis, (b) broken axis, (c) single axis
    fig = plt.figure(figsize=(9.5, 9.0))
    gs = GridSpec(
        3, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.08, hspace=0.55
    )

    # (a) Tracking: broken x-axis
    ax_a_l = fig.add_subplot(gs[0, 0])
    ax_a_r = fig.add_subplot(gs[0, 1], sharey=ax_a_l)
    style_broken_axes(ax_a_l, ax_a_r)

    # Small horizontal separation to avoid marker overlap across the broken seam
    shift_frac = 0.005  # fraction of axis span to shift markers by (0.5%)
    tir_span_left = args.tir_left_max - args.tir_left_min
    tir_shift = shift_frac * tir_span_left
    tir_xs_l = tir_mean - tir_shift
    tir_xs_r = tir_mean + tir_shift

    plot_points_with_errorbars(ax_a_l, tir_xs_l, ys, tir_std, colors=colors)
    plot_points_with_errorbars(ax_a_r, tir_xs_r, ys, tir_std, colors=colors)

    ax_a_l.set_xlim(args.tir_left_min, args.tir_left_max)
    ax_a_r.set_xlim(args.tir_right_min, args.tir_right_max)
    # center title and xlabel across both columns for row (a)
    center_row_title_and_xlabel(fig, ax_a_l, ax_a_r,
                                title="(a) Tracking stability under noise",
                                xlabel="Within-target ratio (%)",
                                title_offset=0.02, xlabel_offset=0.03)

    ax_a_l.set_yticks(ytick_pos)
    ax_a_l.set_yticklabels(ytick_lab)
    ax_a_l.set_ylabel("Method")

    ax_a_l.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax_a_r.grid(True, axis="x", linestyle="--", alpha=0.35)
    add_break_marks(ax_a_l, ax_a_r)

    # (b) Water use: broken x-axis
    ax_b_l = fig.add_subplot(gs[1, 0])
    ax_b_r = fig.add_subplot(gs[1, 1], sharey=ax_b_l)
    style_broken_axes(ax_b_l, ax_b_r)

    # Small horizontal separation to avoid marker overlap across the broken seam
    irr_span_left = args.irr_left_max - args.irr_left_min
    irr_shift = shift_frac * irr_span_left
    irr_xs_l = irr_mean - irr_shift
    irr_xs_r = irr_mean + irr_shift

    plot_points_with_errorbars(ax_b_l, irr_xs_l, ys, irr_std, colors=colors)
    plot_points_with_errorbars(ax_b_r, irr_xs_r, ys, irr_std, colors=colors)

    ax_b_l.set_xlim(args.irr_left_min, args.irr_left_max)
    ax_b_r.set_xlim(args.irr_right_min, args.irr_right_max)
    # center title and xlabel across both columns for row (b)
    center_row_title_and_xlabel(fig, ax_b_l, ax_b_r,
                                title="(b) Irrigation cost under noise",
                                xlabel="Total irrigation (mm)",
                                title_offset=0.02, xlabel_offset=0.03)

    ax_b_l.set_yticks(ytick_pos)
    ax_b_l.set_yticklabels(ytick_lab)
    ax_b_l.set_ylabel("Method")

    ax_b_l.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax_b_r.grid(True, axis="x", linestyle="--", alpha=0.35)
    add_break_marks(ax_b_l, ax_b_r)

    # (c) Stress days: single axis spanning both columns
    ax_c = fig.add_subplot(gs[2, :])
    plot_points_with_errorbars(ax_c, sd_mean, ys, sd_std, colors=colors)
    ax_c.set_title("(c) Stress occurrence under noise", loc="center", pad=8)

    ax_c.set_yticks(ytick_pos)
    ax_c.set_yticklabels(ytick_lab)
    ax_c.set_ylabel("Method")
    ax_c.set_xlabel("Stress days")
    # nudge the xlabel to be centered under the spanning axis
    ax_c.xaxis.set_label_coords(0.5, -0.12)

    ax_c.set_xlim(args.stress_min, args.stress_max)
    ax_c.grid(True, axis="x", linestyle="--", alpha=0.35)

    # Optional footnote (kept minimal to avoid clutter)
    if args.note:
        fig.text(0.02, 0.01, args.note, fontsize=9)

    fig.tight_layout()

    # Save
    out_png = args.out_prefix + ".png"
    out_pdf = args.out_prefix + ".pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table9", type=str, default="Table9_robust_noise_mean_std.csv")

    # Tracking (Within-target ratio, %): broken axis ranges
    p.add_argument("--tir_left_min", type=float, default=0.0)
    p.add_argument("--tir_left_max", type=float, default=2.0)
    p.add_argument("--tir_right_min", type=float, default=90.0)
    p.add_argument("--tir_right_max", type=float, default=100.0)

    # Irrigation (mm): broken axis ranges
    p.add_argument("--irr_left_min", type=float, default=70.0)
    p.add_argument("--irr_left_max", type=float, default=140.0)
    p.add_argument("--irr_right_min", type=float, default=540.0)
    p.add_argument("--irr_right_max", type=float, default=660.0)

    # Stress days axis
    p.add_argument("--stress_min", type=float, default=0.0)
    p.add_argument("--stress_max", type=float, default=1.0)

    p.add_argument("--out_prefix", type=str, default="fig8_noise_robustness_brokenaxis")
    p.add_argument("--note", type=str, default="Error bars: std over seeds (n = 10).")
    args = p.parse_args()
    main(args)
