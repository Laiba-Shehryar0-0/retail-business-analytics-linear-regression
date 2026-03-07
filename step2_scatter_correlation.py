
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("Sample_-_Superstore.csv", encoding="latin1")
NUM_COLS = ["Sales", "Profit", "Discount", "Quantity"]
COLORS   = ["#2E75B6", "#E07B30", "#4CAF50", "#E53935", "#9C27B0", "#00ACC1"]

#  STEP 2B — Scatter Plot Analysis

print("=" * 70)
print("  STEP 2B — SCATTER PLOT ANALYSIS & LINEAR RELATIONSHIP IDENTIFICATION")
print("=" * 70)

# All pairs to examine
PAIRS = [
    # (X column,    Y column,    colour,     relationship type)
    ("Discount", "Sales",    COLORS[0], "Weak Negative"),
    ("Discount", "Profit",   COLORS[3], "Negative"),
    ("Quantity", "Profit",   COLORS[2], "Weak Positive"),
    ("Sales",    "Profit",   COLORS[1], "Positive"),
    ("Sales",    "Quantity", COLORS[4], "Weak Positive"),
    ("Quantity", "Sales",    COLORS[5], "Weak Positive"),
]

print(f"\n  {'X (Independent)':<18} {'Y (Dependent)':<14} "
      f"{'Pearson r':>10} {'R²':>8}  Relationship")
print("  " + "-" * 70)

pair_results = []
for xcol, ycol, color, rel in PAIRS:
    r  = np.corrcoef(df[xcol], df[ycol])[0, 1]
    r2 = r ** 2
    X  = df[xcol].values.reshape(-1, 1)
    lr = LinearRegression().fit(X, df[ycol].values)
    pair_results.append({
        "xcol": xcol, "ycol": ycol, "color": color,
        "r": r, "r2": r2, "rel": rel,
        "coef": lr.coef_[0], "intercept": lr.intercept_
    })
    print(f"  {xcol:<18} {ycol:<14} {r:>10.4f} {r2:>8.4f}  {rel}")

print("\n  Interpretation Guide:")
print("  |r| > 0.7  → Strong linear relationship")
print("  |r| 0.4-0.7→ Moderate linear relationship")
print("  |r| 0.2-0.4→ Weak linear relationship")
print("  |r| < 0.2  → Negligible / no linear relationship")
#  FIGURE 2 — All 6 Scatter Plots with regression lines

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Scatter Plots with Lines of Best Fit (Identifying Linear Relationships)",
             fontsize=15, fontweight="bold")

for ax, pr in zip(axes.flat, pair_results):
    xcol, ycol = pr["xcol"], pr["ycol"]
    color      = pr["color"]

    # Scatter
    ax.scatter(df[xcol], df[ycol], alpha=0.15, s=12, color=color, label="Data points")

    # Regression line
    xline = np.linspace(df[xcol].min(), df[xcol].max(), 300)
    yline = pr["coef"] * xline + pr["intercept"]
    ax.plot(xline, yline, color="black", linewidth=2.5,
            label=f"y = {pr['coef']:.3f}x + {pr['intercept']:.2f}")

    # Stats annotation box
    txt = (f"Pearson r = {pr['r']:.4f}\n"
           f"R²        = {pr['r2']:.4f}\n"
           f"Relation  : {pr['rel']}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      alpha=0.9, edgecolor="gray"))

    ax.set_title(f"{ycol}  vs  {xcol}", fontsize=12, fontweight="bold")
    ax.set_xlabel(xcol, fontsize=11)
    ax.set_ylabel(ycol, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_facecolor("#F7F9FC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/fig2_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 2 saved → outputs/fig2_scatter_plots.png")

# =============================================================================
#  Detailed interpretation for each pair
# =============================================================================

print("\n" + "=" * 70)
print("  DETAILED RELATIONSHIP INTERPRETATIONS")
print("=" * 70)

interpretations = {
    ("Discount", "Sales"):
        ("Weak Negative",
         "X = Discount (independent)  |  Y = Sales (dependent)",
         "Higher discounts do NOT reliably increase sales revenue. "
         "Customers may already intend to buy regardless of discount."),
    ("Discount", "Profit"):
        ("Negative",
         "X = Discount (independent)  |  Y = Profit (dependent)",
         "Higher discounts directly reduce profit margin per transaction. "
         "Every 10% discount cuts expected profit significantly. "
         "This is the most important negative relationship in the dataset."),
    ("Quantity", "Profit"):
        ("Weak Positive",
         "X = Quantity (independent)  |  Y = Profit (dependent)",
         "Selling more units slightly increases profit, but the effect is weak. "
         "Unit count alone does not guarantee profitability — pricing matters more."),
    ("Sales",    "Profit"):
        ("Positive",
         "X = Sales (independent)  |  Y = Profit (dependent)",
         "Higher total sales revenue is the strongest driver of profit. "
         "This is the most actionable positive relationship (r=0.479)."),
    ("Sales",    "Quantity"):
        ("Weak Positive",
         "X = Sales (independent)  |  Y = Quantity (dependent)",
         "Higher revenue orders tend to include more items, "
         "but the relationship is weak — individual high-price items can have low quantity."),
    ("Quantity", "Sales"):
        ("Weak Positive",
         "X = Quantity (independent)  |  Y = Sales (dependent)",
         "More items in an order gives slightly higher sales total, "
         "but product price dominates over unit count."),
}

for (xcol, ycol), (rel, var_desc, biz) in interpretations.items():
    r = np.corrcoef(df[xcol], df[ycol])[0, 1]
    print(f"\n  {ycol} vs {xcol}  (r={r:.4f})")
    print(f"  Relationship : {rel}")
    print(f"  Variables    : {var_desc}")
    print(f"  Business     : {biz}")

# =============================================================================
#  FIGURE 3 — Correlation Heatmap + Bar chart
# =============================================================================

corr = df[NUM_COLS].corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figure 3 — Correlation Analysis", fontsize=15, fontweight="bold")

# Heatmap
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0,
            ax=axes[0], square=True, linewidths=2, linecolor="white",
            annot_kws={"size": 13, "fontweight": "bold"},
            vmin=-1, vmax=1,
            cbar_kws={"label": "Pearson r", "shrink": 0.8})
axes[0].set_title("Pearson Correlation Heatmap", fontsize=13, fontweight="bold")
axes[0].tick_params(labelsize=11)

# Correlation with Profit — bar chart
profit_corr = corr["Profit"].drop("Profit").sort_values()
bar_colors  = [COLORS[3] if v < 0 else COLORS[2] for v in profit_corr]
bars = axes[1].barh(profit_corr.index, profit_corr.values,
                    color=bar_colors, edgecolor="white", height=0.4)
for bar, val in zip(bars, profit_corr.values):
    offset = 0.006 if val >= 0 else -0.006
    ha     = "left"  if val >= 0 else "right"
    axes[1].text(val + offset, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", ha=ha, fontsize=11, fontweight="bold")
axes[1].axvline(0, color="black", linewidth=1.5)
axes[1].set_title("Correlation with Profit", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Pearson r", fontsize=11)
axes[1].set_facecolor("#F7F9FC")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
red_p = mpatches.Patch(color=COLORS[3], label="Negative correlation (bad for profit)")
grn_p = mpatches.Patch(color=COLORS[2], label="Positive correlation (good for profit)")
axes[1].legend(handles=[red_p, grn_p], fontsize=10, loc="lower right")

plt.tight_layout()
plt.savefig("outputs/fig3_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 3 saved → outputs/fig3_correlation.png")

# =============================================================================
#  FIGURE 4 — Seaborn Pairplot
# =============================================================================

print("\n  Generating pairplot (may take a few seconds)...")
g = sns.pairplot(df[NUM_COLS], diag_kind="kde",
                 plot_kws={"alpha": 0.15, "s": 10},
                 diag_kws={"color": COLORS[0], "fill": True})
g.figure.suptitle("Figure 4 — Pairplot: All Numerical Variable Combinations",
                   y=1.02, fontsize=14, fontweight="bold")
plt.savefig("outputs/fig4_pairplot.png", dpi=130, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 4 saved → outputs/fig4_pairplot.png")

print("\n" + "=" * 70)
print("  MEMBER 2 COMPLETE ✅")
print("=" * 70)
