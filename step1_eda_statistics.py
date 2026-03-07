# =============================================================================
#  MEMBER 1  —  Step 1: Dataset Selection  +  Step 2A: Descriptive Statistics
#  Assignment 1 | Linear Regression for Retail Business Analytics | IIUI FCIT
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ── Output folder ─────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# =============================================================================
#  STEP 1 — Load & Describe Dataset
# =============================================================================

print("=" * 65)
print("  STEP 1 — DATASET SELECTION & OVERVIEW")
print("=" * 65)

df = pd.read_csv("Sample_-_Superstore.csv", encoding="latin1")

print(f"\n  Dataset  : Sample Superstore (Kaggle)")
print(f"  Rows     : {df.shape[0]:,}")
print(f"  Columns  : {df.shape[1]}")
print(f"\n  All Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:2}. {col}")

print(f"\n  First 5 rows:")
print(df.head().to_string())

print(f"\n  Data Types & Missing Values:")
info = pd.DataFrame({
    "Dtype"    : df.dtypes,
    "Non-Null" : df.notnull().sum(),
    "Missing"  : df.isnull().sum(),
    "Missing %" : (df.isnull().mean() * 100).round(2),
})
print(info.to_string())

# =============================================================================
#  STEP 2A — Descriptive Statistics
# =============================================================================

print("\n" + "=" * 65)
print("  STEP 2A — DESCRIPTIVE STATISTICS")
print("=" * 65)

NUM_COLS = ["Sales", "Profit", "Discount", "Quantity"]

# ── Build full stats table ────────────────────────────────────────────────────
stats = df[NUM_COLS].describe().T
stats.columns = ["Count", "Mean", "Std Dev", "Min",
                 "Q1 (25%)", "Median", "Q3 (75%)", "Max"]
stats["Mode"]     = df[NUM_COLS].mode().iloc[0]
stats["Variance"] = df[NUM_COLS].var().round(3)
stats["Range"]    = stats["Max"] - stats["Min"]
stats["IQR"]      = stats["Q3 (75%)"] - stats["Q1 (25%)"]
stats["Skewness"] = df[NUM_COLS].skew().round(4)
stats["Kurtosis"] = df[NUM_COLS].kurt().round(4)

DISPLAY_ORDER = ["Count", "Mean", "Median", "Mode", "Std Dev", "Variance",
                 "Min", "Q1 (25%)", "Q3 (75%)", "Max", "Range", "IQR",
                 "Skewness", "Kurtosis"]
stats = stats[DISPLAY_ORDER].round(3)

print("\n  Full Descriptive Statistics Table:")
print(stats.to_string())

print("\n  Interpretation:")
for col in NUM_COLS:
    mean   = df[col].mean()
    median = df[col].median()
    std    = df[col].std()
    skew   = df[col].skew()
    print(f"\n  {col}:")
    print(f"    Mean={mean:.2f}  |  Median={median:.2f}  |  Std={std:.2f}  |  Skew={skew:.2f}")
    if skew > 1:
        print(f"    → Highly right-skewed: a few very large {col} values pull the mean up.")
    elif skew < -1:
        print(f"    → Highly left-skewed: a few very small {col} values pull the mean down.")
    else:
        print(f"    → Approximately symmetric distribution.")

# =============================================================================
#  STEP 2A — Outlier Detection (IQR Method)
# =============================================================================

print("\n" + "=" * 65)
print("  STEP 2A — OUTLIER DETECTION (IQR Method)")
print("=" * 65)
print(f"\n  Rule: flag value if < Q1 − 1.5×IQR  or  > Q3 + 1.5×IQR\n")

outlier_rows = {}
for col in NUM_COLS:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask  = (df[col] < lower) | (df[col] > upper)
    n     = mask.sum()
    pct   = n / len(df) * 100
    outlier_rows[col] = mask
    print(f"  {col:<10}  Q1={Q1:>8.2f}  Q3={Q3:>8.2f}  IQR={IQR:>7.2f}"
          f"  Fences=[{lower:.2f}, {upper:.2f}]"
          f"  Outliers: {n} ({pct:.1f}%)")

print("\n  Note: Outliers retained — they represent valid extreme transactions.")

# =============================================================================
#  FIGURE 1 — Histograms + KDE
# =============================================================================

COLORS = ["#2E75B6", "#E07B30", "#4CAF50", "#E53935"]

fig = plt.figure(figsize=(16, 10))
fig.suptitle("Figure 1 — Distribution of Key Numerical Variables\n"
             "(Histograms with Mean & Median)",
             fontsize=15, fontweight="bold", y=1.01)

gs = gridspec.GridSpec(2, 4, figure=fig)

for i, (col, color) in enumerate(zip(NUM_COLS, COLORS)):
    # Histogram (top row)
    ax = fig.add_subplot(gs[0, i])
    ax.hist(df[col], bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(df[col].mean(),   color="black", lw=2, ls="--",
               label=f"Mean={df[col].mean():.1f}")
    ax.axvline(df[col].median(), color="gray",  lw=2, ls=":",
               label=f"Median={df[col].median():.1f}")
    ax.set_title(col, fontsize=12, fontweight="bold")
    ax.set_xlabel(col); ax.set_ylabel("Frequency")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor("#F7F9FC")
    ax.legend(fontsize=8)

    # Boxplot (bottom row)
    ax2 = fig.add_subplot(gs[1, i])
    bp = ax2.boxplot(df[col], patch_artist=True, notch=False,
                     boxprops=dict(facecolor=color, alpha=0.5),
                     medianprops=dict(color="black", linewidth=2.5),
                     flierprops=dict(marker="o", markerfacecolor=color,
                                     markersize=2, alpha=0.4),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=2))
    ax2.set_title(f"{col} — Boxplot", fontsize=11)
    ax2.set_ylabel(col)
    ax2.set_facecolor("#F7F9FC")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Annotate Q1/Q3/IQR on boxplot
    Q1 = df[col].quantile(0.25); Q3 = df[col].quantile(0.75)
    ax2.text(1.35, Q1, f"Q1={Q1:.1f}", va="center", fontsize=7, color="navy")
    ax2.text(1.35, Q3, f"Q3={Q3:.1f}", va="center", fontsize=7, color="navy")

plt.tight_layout()
plt.savefig("outputs/fig1_distributions_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 1 saved → outputs/fig1_distributions_boxplots.png")

print("\n" + "=" * 65)
print("  MEMBER 1 COMPLETE ✅")
print("=" * 65)
