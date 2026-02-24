import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_nature_style():
    """Clean, restrained, publication-friendly."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "savefig.dpi": 300,
    })


def compute_summary(df: pd.DataFrame, dt_days: float = 1.0) -> dict:
    Dr = df["Dr"].to_numpy(float)
    lo = df["Dr_lo"].to_numpy(float)
    hi = df["Dr_hi"].to_numpy(float)
    I = df["I"].to_numpy(float)

    within = (Dr >= lo) & (Dr <= hi)
    within_ratio = float(np.mean(within))
    stress_days = int(np.sum(Dr > hi))
    under_days = int(np.sum(Dr < lo))

    # Total irrigation in mm over horizon (assume I is mm/day, multiply by dt)
    total_irr = float(np.sum(I) * dt_days)

    return {
        "within_ratio": within_ratio,
        "stress_days": stress_days,
        "under_days": under_days,
        "total_irr_mm": total_irr,
        "I_max": float(np.max(I)),
        "I_p95": float(np.percentile(I, 95)),
        "T": int(len(df)),
        "dt_days": float(dt_days),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, required=True, help="trajectory.csv path")
    ap.add_argument("--out", type=str, default="figures/fig5_overlay", help="output file prefix without suffix")
    ap.add_argument("--skip_first", type=int, default=0, help="skip first N steps in plotting (visual only)")
    ap.add_argument("--dt_days", type=float, default=1.0, help="time step in days (for total irrigation)")
    ap.add_argument("--clip_I", type=float, default=-1.0,
                    help="clip irrigation y-axis at this value (e.g., 2.5). -1 means cap at 95th percentile.")
    ap.add_argument("--title", type=str, default="",
                    help="(Optional) in-figure title. Recommended: leave empty and use LaTeX caption.")
    ap.add_argument("--mark_out", action="store_true", help="mark out-of-target points on Dr curve")
    ap.add_argument("--mark_clipped", action="store_true", help="mark clipped irrigation peaks at cap")
    ap.add_argument("--hide_legend", action="store_true", help="hide legend (useful if caption explains)")
    args = ap.parse_args()

    set_nature_style()

    df = pd.read_csv(args.traj)

    # Basic column check
    required_cols = {"Dr", "Dr_lo", "Dr_hi", "I"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.skip_first > 0:
        df = df.iloc[args.skip_first:].reset_index(drop=True)

    x = df["day"].to_numpy() if "day" in df.columns else np.arange(len(df))
    Dr = df["Dr"].to_numpy(float)
    lo = df["Dr_lo"].to_numpy(float)
    hi = df["Dr_hi"].to_numpy(float)
    I = df["I"].to_numpy(float)

    summ = compute_summary(df, dt_days=args.dt_days)

    # ---- Color palette: restrained ----
    c_dr = "#1f4e79"      # deep blue
    c_band = "#a7c7e7"    # light blue
    c_bar = "#6b7280"     # neutral gray
    c_out = "#b91c1c"     # deep red

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 3.6))

    # (1) Target interval band: [D_lo,t, D_hi,t]
    ax.fill_between(x, lo, hi, color=c_band, alpha=0.28, label=r"Target interval $[D_{lo,t}, D_{hi,t}]$")
    ax.plot(x, lo, linestyle="--", linewidth=1.2, color=c_dr, alpha=0.55)
    ax.plot(x, hi, linestyle="--", linewidth=1.2, color=c_dr, alpha=0.55)

    # (2) Dr curve
    ax.plot(x, Dr, color=c_dr, linewidth=2.2, label=r"$D_{r,t}$")

    # (3) Optional out-of-range marks (kept off by default to stay clean)
    if args.mark_out:
        out_mask = (Dr < lo) | (Dr > hi)
        if np.any(out_mask):
            ax.scatter(x[out_mask], Dr[out_mask], color=c_out, s=18, zorder=6, linewidths=0.0)

    ax.set_xlabel("Day")
    ax.set_ylabel(r"Root-zone depletion $D_{r,t}$ (mm)")

    if args.title.strip():
        ax.set_title(args.title)

    # Secondary axis for irrigation
    ax2 = ax.twinx()
    ax2.set_ylabel(r"Irrigation $I_t$ (mm/day)")

    # Determine cap for visual clipping
    if args.clip_I > 0:
        I_cap = float(args.clip_I)
        cap_mode = "manual"
    else:
        I_cap = max(1e-6, float(summ["I_p95"]))
        cap_mode = "p95"

    I_plot = np.minimum(I, I_cap)
    ax2.bar(x, I_plot, width=0.9, color=c_bar, alpha=0.18, edgecolor="none")

    # Optional: mark clipped peaks (off by default)
    if args.mark_clipped:
        clipped = I > I_cap
        if np.any(clipped):
            ax2.scatter(x[clipped], np.full(np.sum(clipped), I_cap),
                        color=c_bar, s=14, zorder=7)

    # Summary box: keep concise + clearly "one rollout"
    txt = (
        "Representative rollout\n"
        f"Within-target: {summ['within_ratio']*100:.2f}%\n"
        f"Total irrigation: {summ['total_irr_mm']:.2f} mm\n"
        f"Stress / Under: {summ['stress_days']} / {summ['under_days']}"
    )
    ax.text(
        0.72, 0.26, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#dddddd", alpha=0.86),
    )

    # Aesthetics
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Legend: minimal and clean
    if not args.hide_legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, l, loc="lower left", frameon=True, framealpha=0.95)

    fig.tight_layout()

    # Save figure
    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)

    # Save a summary file for caption consistency
    summary_payload = {
        "note": "Values are from ONE representative rollout (one seed/one episode), not mean±std.",
        "cap_mode": cap_mode,
        "I_cap_view": float(I_cap),
        "I_max_actual": float(summ["I_max"]),
        **summ,
        "skip_first_visual_only": int(args.skip_first),
    }
    (out_prefix.parent).mkdir(parents=True, exist_ok=True)
    with open(out_prefix.with_name(out_prefix.name + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(out_prefix.with_name(out_prefix.name + "_summary.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Representative rollout (one seed; one episode; nominal setting).\n"
            f"Within-target = {summ['within_ratio']*100:.2f}% ; "
            f"Total irrigation = {summ['total_irr_mm']:.2f} mm ; "
            f"Stress/Under = {summ['stress_days']}/{summ['under_days']}.\n"
            f"Visualization: irrigation axis capped at I_cap_view = {I_cap:.3f} mm/day "
            f"({cap_mode}); actual max I_max = {summ['I_max']:.3f} mm/day.\n"
            f"Visual-only: skip_first = {args.skip_first} steps.\n"
        )

    print(f"[OK] Saved: {out_prefix.with_suffix('.png')} and {out_prefix.with_suffix('.pdf')}")
    print(f"[OK] Saved: {out_prefix.with_name(out_prefix.name + '_summary.json')}")
    print(f"[OK] Saved: {out_prefix.with_name(out_prefix.name + '_summary.txt')}")


if __name__ == "__main__":
    main()
