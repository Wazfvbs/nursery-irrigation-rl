import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib.patches import Patch

def _get_x(df):
    return df["day"].to_numpy() if "day" in df.columns else np.arange(len(df))

def _col(df, name, default=0.0):
    return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), float(default))

def panel_label(ax, text):
    ax.text(0.01, 0.98, text, transform=ax.transAxes,
            va="top", ha="left", fontsize=12, fontweight="bold")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", type=str, required=True)
    ap.add_argument("--fao", type=str, required=True)
    ap.add_argument("--thr", type=str, required=True)
    ap.add_argument("--out", type=str, default="fig6_final.pdf")
    ap.add_argument("--title", type=str, default="Fig.6 Baseline failure: over-irrigation and target misalignment (Nominal)")
    ap.add_argument("--skip_days", type=int, default=3, help="Skip first N days (remove initial transient)")
    ap.add_argument("--clip_I", type=float, default=2.5, help="Clip PPO irrigation plot for visibility")
    args = ap.parse_args()

    # -------- load & align --------
    ppo = pd.read_csv(args.ppo)
    fao = pd.read_csv(args.fao)
    thr = pd.read_csv(args.thr)

    T = min(len(ppo), len(fao), len(thr))
    ppo = ppo.iloc[:T].reset_index(drop=True)
    fao = fao.iloc[:T].reset_index(drop=True)
    thr = thr.iloc[:T].reset_index(drop=True)

    s = max(int(args.skip_days), 0)

    x = _get_x(ppo)[s:]

    Dr_ppo = _col(ppo, "Dr")[s:]
    Dr_fao = _col(fao, "Dr")[s:]
    Dr_thr = _col(thr, "Dr")[s:]

    Dr_lo = _col(ppo, "Dr_lo")[s:]
    Dr_hi = _col(ppo, "Dr_hi")[s:]

    I_ppo = _col(ppo, "I")[s:]
    I_fao = _col(fao, "I")[s:]
    I_thr = _col(thr, "I")[s:]

    # totals
    sum_ppo = float(np.sum(I_ppo))
    sum_fao = float(np.sum(I_fao))
    sum_thr = float(np.sum(I_thr))
    baseline_avg = 0.5 * (sum_fao + sum_thr)

    saving_vs_avg = (1.0 - sum_ppo / baseline_avg) * 100.0 if baseline_avg > 0 else 0.0
    saving_vs_fao = (1.0 - sum_ppo / sum_fao) * 100.0 if sum_fao > 0 else 0.0
    saving_vs_thr = (1.0 - sum_ppo / sum_thr) * 100.0 if sum_thr > 0 else 0.0

    # -------- style --------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
    })

    # 更高级且色盲友好的配色（论文常用）
    c_ppo = "#0072B2"   # deep blue
    c_fao = "#D55E00"   # vermillion/orange-red
    c_thr = "#6E6E6E"   # dark gray
    c_band = "#56B4E9"  # light blue for band

    # -------- layout: 3 rows --------
    fig = plt.figure(figsize=(9.2, 6.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 1.6, 1.4], hspace=0.40)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # =========================
    # (a) Dr distribution (boxplot) + target band
    # =========================
    lo_ref = float(np.median(Dr_lo))
    hi_ref = float(np.median(Dr_hi))

    ax1.axhspan(lo_ref, hi_ref, alpha=0.18, color=c_band)

    data = [Dr_ppo, Dr_fao, Dr_thr]
    labels = ["PPO", "FAO Rule", "Threshold"]

    bp = ax1.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops=dict(linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    colors = [c_ppo, c_fao, c_thr]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_linewidth(1.2)

    for med, col in zip(bp["medians"], colors):
        med.set_color(col)

    ax1.set_title("Dr distribution: PPO stays inside target band, baselines collapse near 0 (over-irrigation)")
    ax1.set_ylabel("Root-zone depletion Dr (mm)")
    ax1.grid(True, alpha=0.22)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    panel_label(ax1, "(a)")

    # legend for target band
    ax1.legend(
        handles=[Patch(facecolor=c_band, edgecolor="none", alpha=0.25, label=f"Target band ~ [{lo_ref:.1f}, {hi_ref:.1f}] mm")],
        loc="upper right",
        frameon=False
    )

    # =========================
    # (b) irrigation actions: PPO clipped + inset baselines full scale
    # =========================
    ax2.step(x, I_ppo, where="mid", color=c_ppo, linewidth=2.4, label="PPO (clipped view)")
    ax2.set_ylim(0, float(args.clip_I))
    ax2.set_title(f"Irrigation strategy: PPO is low & smooth (clipped to {args.clip_I:.1f} mm/day)")
    ax2.set_ylabel("I (mm/day)")
    ax2.grid(True, alpha=0.22)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper left", frameon=False)
    panel_label(ax2, "(b)")

    # inset for baselines
    # inset for baselines (full scale) - BAR SPIKES
    inset = ax2.inset_axes([0.60, 0.18, 0.38, 0.75])

    mask_fao = I_fao > 1e-9
    mask_thr = I_thr > 1e-9

    w = 0.35  # bar width (small to avoid clutter)

    # FAO bars (slightly left)
    inset.vlines(x[mask_fao], 0, I_fao[mask_fao], color=c_fao, alpha=0.55, linewidth=1.2, label="FAO Rule")
    inset.vlines(x[mask_thr], 0, I_thr[mask_thr], color=c_thr, alpha=0.45, linewidth=1.2, label="Threshold")


    inset.set_title("Baselines (full scale)", fontsize=9)

    # 更干净：弱化网格 / 右上角边框
    inset.grid(True, alpha=0.15)
    inset.tick_params(labelsize=8)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)

    # 合理的 y 轴范围（避免太空/太挤）
    ymax = max(float(np.max(I_fao)), float(np.max(I_thr)), 1.0)
    inset.set_ylim(0, ymax * 1.05)
    inset.set_xlim(x[int(len(x)*0.25)], x[-1])
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.95)


# =========================
    # (c) total irrigation bar + saving ratio
    # =========================
    methods = ["PPO", "FAO Rule", "Threshold"]
    totals = [sum_ppo, sum_fao, sum_thr]
    cols = [c_ppo, c_fao, c_thr]

    x3 = np.arange(len(methods))
    bars = ax3.bar(x3, totals, color=cols, alpha=0.85)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel("Total irrigation ΣI (mm)")
    ax3.set_title("Total irrigation and water-saving ratio (baseline over-irrigation)")
    ax3.grid(True, axis="y", alpha=0.22)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    panel_label(ax3, "(c)")

    # annotate values on bars
    for b, val in zip(bars, totals):
        ax3.text(b.get_x() + b.get_width()/2, val + 5, f"{val:.1f}",
                 ha="center", va="bottom", fontsize=9)

    # add saving text
    ax3.text(
        0.02, 0.92,
        f"Water saving vs FAO: {saving_vs_fao:.1f}%  |  vs Threshold: {saving_vs_thr:.1f}%\n"
        f"Water saving vs baseline avg: {saving_vs_avg:.1f}%",
        transform=ax3.transAxes,
        fontsize=9,
        va="top"
    )

    ax3.set_xlabel("Method")

    # overall title
    fig.suptitle(args.title, y=1.01, fontsize=14)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] saved: {out} and {out.with_suffix('.png')}")

if __name__ == "__main__":
    main()
