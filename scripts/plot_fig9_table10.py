import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Load the data again
table_path = "../outputs/tables/Table10_ablation_mean_std.csv"
df = pd.read_csv(table_path)

# Parse mean ± std from string
def parse_mean_std(s):
    parts = re.split(r"\s*±\s*", str(s))
    mean = float(parts[0])
    std = float(parts[1]) if len(parts) > 1 else 0.0
    return mean, std

# Plotting grouped bar chart with enhanced readability
def plot_grouped_bar(ax, categories, values, stds, colors, ylabel, title):
    x = np.arange(len(categories))
    bars = ax.bar(x, values, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)

    # Add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=12)

    ax.set_ylim([0, max(values) * 1.5])  # Increase y-axis limit for better spacing

# Prepare data for Fig.9.a (Full vs w/o_UCB)
methods_a = ['Full', 'wo_UCB']
categories_a = ['WithinTargetRatio', 'TotalIrrigation_mm', 'StressDays', 'MAE_mm', 'RMSE_mm']

values_a = {col: [] for col in categories_a}
stds_a = {col: [] for col in categories_a}

for m in methods_a:
    row = df[df["case"] == m].iloc[0]
    for col in categories_a:
        mean, std = parse_mean_std(row[col])
        values_a[col].append(mean)
        stds_a[col].append(std)

# Fig.9.a: Full vs w/o_UCB - Grouped Bar Chart with improved y-axis
fig, axes = plt.subplots(2, 3, figsize=(22, 14), sharey=False)  # Adjusted size for better space utilization
axes = axes.ravel()
colors_a = ["#ADD8E6", "#4682B4"]  # Light Blue for Full, Dark Blue for w/o_UCB

# Adjust y limits dynamically based on data for each subplot
for i, col in enumerate(categories_a):
    ax = axes[i]
    plot_grouped_bar(ax, methods_a, values_a[col], stds_a[col], colors_a, col, f"Full vs wo_UCB - {col}")

    # Adjusting y-limits for each plot based on max value
    y_max = max(values_a[col]) + max(stds_a[col]) * 1.2
    ax.set_ylim([0, y_max])

# Tight layout adjustment
fig.tight_layout(pad=2.5)
plt.savefig("../outputs/figures/fig9_a_comparison_optimized_v4.png", dpi=300)
plt.close()

"../outputs/figures/fig9_a_comparison_optimized_v4.png"
