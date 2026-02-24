import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# -----------------------------
# Utilities
# -----------------------------
def parse_mean_std(x):
    """Parse 'mean ± std' or numeric."""
    if isinstance(x, (int, float, np.floating)):
        return float(x), 0.0
    s = str(x).strip()
    parts = re.split(r"\s*±\s*", s)
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std


def load_stats_table(csv_path: str, key_col: str = "method"):
    df = pd.read_csv(csv_path)
    df = df.copy()
    df[key_col] = df[key_col].astype(str)

    rec = {}
    for _, row in df.iterrows():
        key = row[key_col]
        metrics = {}
        for col in df.columns:
            if col == key_col:
                continue
            m, s = parse_mean_std(row[col])
            metrics[col] = (m, s)
        rec[key] = metrics
    return rec


def rename_methods(methods_raw):
    """Academic naming for paper."""
    mapping = {
        "FAORule": "FAO-rule",
        "Threshold": "Threshold-rule",
    }
    return [mapping.get(m, m) for m in methods_raw]


def build_display_stats(raw_stats, methods_raw):
    """Convert dict keyed by raw names -> dict keyed by display names."""
    methods_display = rename_methods(methods_raw)
    mapping = dict(zip(methods_raw, methods_display))
    out = {}
    for raw in methods_raw:
        out[mapping[raw]] = raw_stats[raw]
    return out, methods_display


def pick_colors(methods_display):
    """
    Simple, clean palette. PPO emphasized.
    """
    palette = {
        "PPO": "#1f77b4",
        "Threshold-rule": "#ff7f0e",
        "FAO-rule": "#2ca02c",
    }
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    cmap = {}
    for i, m in enumerate(methods_display):
        cmap[m] = palette.get(m, default_cycle[i % len(default_cycle)] if default_cycle else "black")
    return cmap


# -----------------------------
# Plot primitives
# -----------------------------
def dumbbell(ax, methods_display, nominal, noise, metric,
             xlim=None, is_ratio=False, emphasize="PPO",
             show_error=True, error_one_sided=False,
             grid_alpha=0.22):
    """
    Dumbbell chart: nominal (filled) vs noise (hollow).
    """
    y = np.arange(len(methods_display))[::-1]  # top to bottom
    colors = pick_colors(methods_display)

    for i, m in enumerate(methods_display):
        yn = y[i]
        (m0, s0) = nominal[m][metric]
        (m1, s1) = noise[m][metric]

        c = colors[m]
        lw = 2.7 if m == emphasize else 1.7
        ms = 9 if m == emphasize else 7

        # connect line
        ax.plot([m0, m1], [yn, yn], linewidth=lw, alpha=0.85, color=c, zorder=2)

        # points
        ax.plot(m0, yn, marker="o", markersize=ms, color=c, zorder=3)  # nominal
        ax.plot(m1, yn, marker="o", markersize=ms,
                markerfacecolor="none", markeredgewidth=2.1, color=c, zorder=3)  # noise

        if show_error:
            alpha_err = 0.22
            if error_one_sided:
                # xerr shape must be (2, N): lower, upper
                xerr0 = np.array([[0.0], [max(0.0, s0)]])
                xerr1 = np.array([[0.0], [max(0.0, s1)]])
                ax.errorbar(m0, yn, xerr=xerr0, fmt="none", capsize=3, alpha=alpha_err, color=c, zorder=1)
                ax.errorbar(m1, yn, xerr=xerr1, fmt="none", capsize=3, alpha=alpha_err, color=c, zorder=1)
            else:
                ax.errorbar(m0, yn, xerr=s0, fmt="none", capsize=3, alpha=alpha_err, color=c, zorder=1)
                ax.errorbar(m1, yn, xerr=s1, fmt="none", capsize=3, alpha=alpha_err, color=c, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(methods_display)

    ax.grid(True, axis="x", linestyle="--", alpha=grid_alpha)
    ax.set_axisbelow(True)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if is_ratio:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))


def pareto_main(ax, methods_display, nominal, noise,
                x_metric="TotalIrrigation_mm", y_metric="WithinTargetRatio",
                emphasize="PPO"):
    """
    Main Pareto: show all methods, annotate only PPO to avoid clutter.
    """
    colors = pick_colors(methods_display)

    for m in methods_display:
        (x0, sx0) = nominal[m][x_metric]
        (y0, sy0) = nominal[m][y_metric]
        (x1, sx1) = noise[m][x_metric]
        (y1, sy1) = noise[m][y_metric]

        c = colors[m]
        ms = 150 if m == emphasize else 90
        lw = 2.7 if m == emphasize else 1.6

        ax.plot([x0, x1], [y0, y1], linewidth=lw, alpha=0.80, color=c, zorder=2)
        ax.scatter([x0], [y0], s=ms, marker="o", color=c, zorder=3)  # nominal
        ax.scatter([x1], [y1], s=ms, marker="o",
                   facecolors="none", edgecolors=c, linewidths=2.1, zorder=3)  # noise

        # annotate only PPO in main panel
        if m == emphasize:
            ax.text(x0 + 6, y0 - 0.015, f"{m}", fontsize=10, alpha=0.95)

    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.set_axisbelow(True)


def pareto_inset(ax_in, baseline_methods, nominal, noise,
                 x_metric="TotalIrrigation_mm", y_metric="WithinTargetRatio"):
    """
    Baseline zoom for Pareto inset: annotate baselines here.
    """
    colors = pick_colors(["PPO"] + baseline_methods)

    for m in baseline_methods:
        (x0, sx0) = nominal[m][x_metric]
        (y0, sy0) = nominal[m][y_metric]
        (x1, sx1) = noise[m][x_metric]
        (y1, sy1) = noise[m][y_metric]
        c = colors[m]

        ax_in.plot([x0, x1], [y0, y1], linewidth=1.6, alpha=0.85, color=c, zorder=2)
        ax_in.scatter([x0], [y0], s=70, marker="o", color=c, zorder=3)
        ax_in.scatter([x1], [y1], s=70, marker="o",
                      facecolors="none", edgecolors=c, linewidths=2.0, zorder=3)

        # small label offset to avoid overlap
        dx = 2.0
        dy = 0.0006 if "Threshold" in m else -0.0006
        ax_in.text(x0 + dx, y0 + dy, m, fontsize=9, alpha=0.95, va="center")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table8", type=str, required=True,
                    help="Table8_main_results_mean_std.csv (Nominal)")
    ap.add_argument("--table9", type=str, required=True,
                    help="Table9_robust_noise_mean_std.csv (Noise)")
    ap.add_argument("--methods", type=str, default="PPO,Threshold,FAORule",
                    help="comma-separated raw method names")
    ap.add_argument("--out", type=str, default="figures/fig8_robustness_final_v2.png")
    args = ap.parse_args()

    methods_raw = [m.strip() for m in args.methods.split(",") if m.strip()]

    t8_raw = load_stats_table(args.table8, key_col="method")
    t9_raw = load_stats_table(args.table9, key_col="method")

    # sanity check
    for m in methods_raw:
        if m not in t8_raw:
            raise KeyError(f"Missing method '{m}' in Table8: {args.table8}")
        if m not in t9_raw:
            raise KeyError(f"Missing method '{m}' in Table9: {args.table9}")

    nominal, methods_display = build_display_stats(t8_raw, methods_raw)
    noise, _ = build_display_stats(t9_raw, methods_raw)

    # baseline methods (display)
    baseline_methods = [m for m in methods_display if m != "PPO"]

    # style
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
    })

    fig = plt.figure(figsize=(12.5, 8.0))
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.30)

    # -----------
    # (a) Tracking stability
    # -----------
    axa = fig.add_subplot(gs[0, 0])
    axa.set_title("(a) Tracking stability under noise", loc="center", fontweight="bold")

    dumbbell(
        axa, methods_display, nominal, noise,
        metric="WithinTargetRatio",
        xlim=(0.95, 1.00),
        is_ratio=True,
        emphasize="PPO",
        show_error=True
    )
    axa.set_xlabel("Within-target ratio")
    axa.text(0.80, -0.18, "Baselines are zoomed in the inset (0–2%).",
             transform=axa.transAxes, fontsize=9, alpha=0.75)

    # inset for baselines (0–2%)
    axa_in = inset_axes(axa, width="52%", height="42%", loc="lower left", borderpad=1.3)
    if baseline_methods:
        nominal_b = {k: nominal[k] for k in baseline_methods}
        noise_b = {k: noise[k] for k in baseline_methods}

        dumbbell(
            axa_in, baseline_methods, nominal_b, noise_b,
            metric="WithinTargetRatio",
            xlim=(0.00, 0.02),
            is_ratio=True,
            emphasize="",
            show_error=False
        )
        axa_in.set_title("Baseline zoom (0–2%)", fontsize=9, loc="center")
        # ✅ remove y labels to avoid overlap with main panel
        axa_in.set_yticklabels([])
        axa_in.set_ylabel("")
        axa_in.tick_params(axis="y", length=0)
        axa_in.tick_params(axis="x", labelsize=8)
        axa_in.grid(True, axis="x", linestyle="--", alpha=0.20)

    # -----------
    # (b) Irrigation cost
    # -----------
    axb = fig.add_subplot(gs[0, 1])
    axb.set_title("(b) Irrigation cost under noise", loc="center", fontweight="bold")

    dumbbell(
        axb, methods_display, nominal, noise,
        metric="TotalIrrigation_mm",
        xlim=None,
        is_ratio=False,
        emphasize="PPO",
        show_error=True
    )
    axb.set_xlabel("Total irrigation (mm)")

    # PPO inset (smaller, not dominant)
    axb_in = inset_axes(axb, width="46%", height="38%", loc="lower left", borderpad=1.3)
    if "PPO" in methods_display:
        nominal_p = {"PPO": nominal["PPO"]}
        noise_p = {"PPO": noise["PPO"]}
        dumbbell(
            axb_in, ["PPO"], nominal_p, noise_p,
            metric="TotalIrrigation_mm",
            xlim=(70, 130),
            is_ratio=False,
            emphasize="PPO",
            show_error=True
        )
        axb_in.set_title("PPO zoom (70–130mm)", fontsize=9, loc="left")
        axb_in.set_yticklabels([])
        axb_in.set_ylabel("")
        axb_in.tick_params(axis="y", length=0)
        axb_in.tick_params(axis="x", labelsize=8)
        axb_in.grid(True, axis="x", linestyle="--", alpha=0.20)

    # -----------
    # (c) Stress occurrence (non-negative + one-sided error)
    # -----------
    axc = fig.add_subplot(gs[1, 0])
    axc.set_title("(c) Stress occurrence under noise", loc="center", fontweight="bold")

    dumbbell(
        axc, methods_display, nominal, noise,
        metric="StressDays",
        xlim=None,
        is_ratio=False,
        emphasize="PPO",
        show_error=True,
        error_one_sided=True  # ✅ one-sided errorbar
    )
    axc.set_xlabel("Stress days (#, non-negative)")

    # enforce non-negative axis and reasonable max
    max_stress = 0.0
    for m in methods_display:
        max_stress = max(
            max_stress,
            nominal[m]["StressDays"][0] + nominal[m]["StressDays"][1],
            noise[m]["StressDays"][0] + noise[m]["StressDays"][1],
            )
    axc.set_xlim(0.0, max(1.0, max_stress * 1.25))

    # -----------
    # (d) Pareto view + baseline inset
    # -----------
    axd = fig.add_subplot(gs[1, 1])
    axd.set_title("(d) Pareto view (stability vs cost)", loc="center", fontweight="bold")

    pareto_main(
        axd, methods_display, nominal, noise,
        x_metric="TotalIrrigation_mm",
        y_metric="WithinTargetRatio",
        emphasize="PPO"
    )

    axd.set_xlabel("Total irrigation (mm)  ↓ better")
    axd.set_ylabel("Within-target ratio  ↑ better")

    # inset: baseline cluster zoom
    axd_in = inset_axes(axd, width="56%", height="44%", loc="lower right", borderpad=1.3)
    axd_in.set_title("Baseline zoom", fontsize=9, loc="left")
    axd_in.grid(True, linestyle="--", alpha=0.20)
    axd_in.set_axisbelow(True)

    # Set baseline zoom bounds (safe defaults for your results)
    axd_in.set_xlim(540, 660)
    axd_in.set_ylim(0.00, 0.02)
    axd_in.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axd_in.tick_params(axis="both", labelsize=8)

    if baseline_methods:
        pareto_inset(axd_in, baseline_methods, nominal, noise)

    # subtle connectors (lighter to avoid visual dominance)
    mark_inset(axd, axd_in, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.35)

    # -----------
    # Unified legend (Nominal vs Noise) - lower to reduce top whitespace
    # -----------
    h_nom = plt.Line2D([], [], marker='o', linestyle='None', markersize=7,
                       markerfacecolor='black', markeredgecolor='black', label='Nominal')
    h_noise = plt.Line2D([], [], marker='o', linestyle='None', markersize=7,
                         markerfacecolor='none', markeredgecolor='black', label='Noise')

    fig.legend(handles=[h_nom, h_noise],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.52, 0.995))

    # Reduce top whitespace while keeping legend visible
    fig.subplots_adjust(top=0.90)

    # Save
    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")

    plt.savefig(out_png, dpi=600)
    plt.savefig(out_pdf)
    print(f"[OK] Saved PNG: {out_png.resolve()}")
    print(f"[OK] Saved PDF: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
