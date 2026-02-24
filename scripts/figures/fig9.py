# -*- coding: utf-8 -*-
import re
import math
import pandas as pd
import matplotlib.pyplot as plt

# Use Type 42 (TrueType) font embedding in PDFs for better compatibility with LaTeX editors
plt.rcParams['pdf.fonttype'] = 42

# ----------------------------
# Config
# ----------------------------
MAIN_CSV = "../../outputs/tables_ref/Table10_ablation_mean_std.csv"
SUPP_CSV = "../../outputs/tables_ref/Table10_ablation_supp_mean_std.csv"

OUT_MAIN = "Fig9_ablation_main.png"
OUT_SUPP = "Fig9_ablation_supp.png"

# Case display names (edit if your CSV uses different case keys)
CASE_ORDER = ["wo_Shaping", "wo_Target", "wo_UCB"]  # ablations only
CASE_LABEL = {
    "wo_Shaping": "w/o shaping",
    "wo_Target": "w/o target",
    "wo_UCB": "w/o UCB",
}

# If True: convert metrics to "improvement vs Full (higher is better)"
# e.g., Water saving = -(Δ irrigation), Stress reduction = -(Δ stress), Smoothness gain = -(Δ actionTV)
USE_IMPROVEMENT_SIGN = False

# ----------------------------
# Helpers
# ----------------------------
def parse_pm(x):
    """
    Parse string like '100.346310 ± 6.026138' into (mean, std) floats.
    """
    if pd.isna(x):
        return (float("nan"), float("nan"))
    s = str(x).strip()
    parts = re.split(r"\s*±\s*", s)
    if len(parts) == 2:
        return (float(parts[0].strip()), float(parts[1].strip()))
    # fallback: try split by '+/-'
    parts = re.split(r"\s*\+/-\s*", s)
    if len(parts) == 2:
        return (float(parts[0].strip()), float(parts[1].strip()))
    raise ValueError(f"Cannot parse mean±std from: {x}")

def build_mean_std(df, col, case_col="case"):
    """
    Return dict: case -> (mean, std)
    """
    out = {}
    for _, row in df.iterrows():
        case = row[case_col]
        mean, std = parse_pm(row[col])
        out[case] = (mean, std)
    return out

def delta_vs_full(ms_dict, metric_name):
    """
    Compute delta (ablation - full) and propagated std: sqrt(std_a^2 + std_full^2)
    """
    if "Full" not in ms_dict:
        raise KeyError(f"[{metric_name}] missing 'Full' row in CSV.")
    full_mean, full_std = ms_dict["Full"]

    xs, ys, es = [], [], []
    for c in CASE_ORDER:
        if c not in ms_dict:
            raise KeyError(f"[{metric_name}] missing '{c}' row in CSV.")
        m, s = ms_dict[c]
        d = m - full_mean
        e = math.sqrt(s * s + full_std * full_std)
        xs.append(CASE_LABEL.get(c, c))
        ys.append(d)
        es.append(e)
    return xs, ys, es

def maybe_flip(metric_key, ys, es):
    """
    Convert to 'improvement (higher better)' if configured.
    """
    if not USE_IMPROVEMENT_SIGN:
        return ys, es

    # Define which metrics should be negated to make "higher is better"
    # Tracking (TIR) -> higher better (keep)
    # Irrigation, StressDays, UnderDays, ActionTV, ActionStd, MAE, RMSE -> lower better (negate)
    negate = metric_key in {
        "TotalIrrigation_mm", "StressDays_ref", "UnderDays_ref",
        "ActionTV", "ActionStd",
        "MAE_ref_mm", "RMSE_ref_mm"
    }
    if negate:
        ys = [-v for v in ys]
        # errors unchanged under sign flip
    return ys, es

# ----------------------------
# Load tables
# ----------------------------
main_df = pd.read_csv(MAIN_CSV)
supp_df = pd.read_csv(SUPP_CSV)

# ----------------------------
# Main figure (recommended for main text)
# ----------------------------
# Metrics from Table10 main:
#   TIR_ref, TotalIrrigation_mm, StressDays_ref, ActionTV
metrics_main = [
    ("TIR_ref",              r"$\Delta$ within-target (pp)",            True),   # convert ratio to percentage points
    ("TotalIrrigation_mm",   r"$\Delta$ total irrigation (mm)",         False),
    ("StressDays_ref",       r"$\Delta$ stress days (days)",            False),
    ("ActionTV",             r"$\Delta$ action TV",                     False),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes = axes.flatten()

for ax, (key, ylabel, to_pp) in zip(axes, metrics_main):
    ms = build_mean_std(main_df, key)
    xlab, ys, es = delta_vs_full(ms, key)

    # Convert ratio deltas to percentage points if needed
    if to_pp:
        ys = [v * 100.0 for v in ys]
        es = [v * 100.0 for v in es]

    ys, es = maybe_flip(key, ys, es)

    ax.bar(xlab, ys, yerr=es, capsize=4)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(OUT_MAIN, dpi=300, bbox_inches='tight')
fig.savefig(OUT_MAIN.replace('.png', '.pdf'), bbox_inches='tight')
plt.close(fig)

# ----------------------------
# Supplementary diagnostics figure (optional)
# ----------------------------
# Metrics from Table10 supp:
#   MAE_ref_mm, RMSE_ref_mm, UnderDays_ref, ActionStd
metrics_supp = [
    ("MAE_ref_mm",      r"$\Delta$ MAE (mm)",               False),
    ("RMSE_ref_mm",     r"$\Delta$ RMSE (mm)",              False),
    ("UnderDays_ref",   r"$\Delta$ under-days (days)",      False),
    ("ActionStd",       r"$\Delta$ action std",             False),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes = axes.flatten()

for ax, (key, ylabel, _) in zip(axes, metrics_supp):
    ms = build_mean_std(supp_df, key)
    xlab, ys, es = delta_vs_full(ms, key)
    ys, es = maybe_flip(key, ys, es)

    ax.bar(xlab, ys, yerr=es, capsize=4)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(OUT_SUPP, dpi=300, bbox_inches='tight')
fig.savefig(OUT_SUPP.replace('.png', '.pdf'), bbox_inches='tight')
plt.close(fig)

print("Saved:", OUT_MAIN, "and", OUT_SUPP)
