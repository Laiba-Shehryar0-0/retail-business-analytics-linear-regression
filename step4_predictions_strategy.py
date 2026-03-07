# =============================================================================
#  MEMBER 4  —  Step 4: Predictions  +  Step 5: Business Strategy
#  Assignment 1 | Linear Regression for Retail Business Analytics | IIUI FCIT
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, pickle

os.makedirs("outputs", exist_ok=True)

df     = pd.read_csv("Sample_-_Superstore.csv", encoding="latin1")
COLORS = ["#2E75B6", "#E07B30", "#4CAF50", "#E53935", "#9C27B0", "#00ACC1"]

# Load trained models from step 3
with open("outputs/models.pkl", "rb") as f:
    models = pickle.load(f)
m1, m2, m3, m4 = models["m1"], models["m2"], models["m3"], models["m4"]

def predict(model, value):
    return model["coef"] * value + model["intercept"]

# =============================================================================
#  STEP 4 — PREDICTION TASKS
# =============================================================================

print("=" * 65)
print("  STEP 4 — FUTURE PREDICTION TASKS")
print("=" * 65)

# ── SCENARIO A: Profit from Sales ────────────────────────────────────────────
print("\n  SCENARIO A — Predict Profit from Sales Value")
print(f"  Model  : Profit = {m1['coef']:.4f} × Sales + ({m1['intercept']:.4f})")
print(f"  R²     : {m1['r2_test']:.4f}")
print(f"  {'Sales ($)':>12} {'Predicted Profit ($)':>22} {'Status':>10}")
print("  " + "-" * 50)

sales_scenarios = [50, 100, 250, 500, 750, 1000, 2000, 5000, 10000]
pred_a = []
for s in sales_scenarios:
    p      = predict(m1, s)
    status = "✅ Profit" if p > 0 else "❌ Loss"
    pred_a.append((s, p, status))
    print(f"  {s:>12,.0f} {p:>22.2f} {status:>10}")

# ── SCENARIO B: Profit from Discount ─────────────────────────────────────────
print("\n  SCENARIO B — Predict Profit from Discount Level")
breakeven = -m2["intercept"] / m2["coef"]
print(f"  Model       : Profit = {m2['coef']:.4f} × Discount + {m2['intercept']:.4f}")
print(f"  Break-even  : {breakeven*100:.1f}% — discounts above this lead to average losses")
print(f"  {'Discount':>12} {'Predicted Profit ($)':>22} {'Status':>10}")
print("  " + "-" * 50)

disc_scenarios = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
pred_b = []
for d in disc_scenarios:
    p      = predict(m2, d)
    status = "✅ Profit" if p > 0 else "❌ Loss"
    pred_b.append((d, p, status))
    print(f"  {d*100:>11.0f}% {p:>22.2f} {status:>10}")

# ── SCENARIO C: Profit from Quantity ─────────────────────────────────────────
print("\n  SCENARIO C — Predict Profit from Quantity Sold")
print(f"  Model  : Profit = {m3['coef']:.4f} × Quantity + {m3['intercept']:.4f}")
print(f"  {'Quantity':>12} {'Predicted Profit ($)':>22}")
print("  " + "-" * 38)

qty_scenarios = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
pred_c = []
for q in qty_scenarios:
    p = predict(m3, q)
    pred_c.append((q, p))
    print(f"  {q:>12} {p:>22.2f}")

# ── SCENARIO D: Combined realistic business cases ────────────────────────────
print("\n  SCENARIO D — 5 Realistic Business Prediction Cases")
print("  " + "-" * 65)
cases = [
    ("Low-value order, no discount",   "Sales",    150,  m1),
    ("Mid-value order, 20% discount",  "Discount", 0.20, m2),
    ("High-value order",               "Sales",    2500, m1),
    ("Bulk quantity (10 units)",       "Quantity", 10,   m3),
    ("Heavy discount (45%)",           "Discount", 0.45, m2),
]
for desc, var, val, mod in cases:
    p      = predict(mod, val)
    status = "✅ Profit" if p > 0 else "❌ Loss"
    print(f"  {desc:<40}  {var}={val}  →  Profit=${p:.2f}  {status}")

# =============================================================================
#  FIGURE 10 — Prediction Visualizations (3 panels)
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Figure 10 — Prediction Scenarios", fontsize=15, fontweight="bold")

# Panel A: Profit from Sales
ax = axes[0]
s_vals = [s for s, _, _ in pred_a]
p_vals = [p for _, p, _ in pred_a]
bar_c  = [COLORS[2] if p >= 0 else COLORS[3] for p in p_vals]
bars   = ax.bar(range(len(s_vals)), p_vals, color=bar_c, edgecolor="white", width=0.6)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(range(len(s_vals)))
ax.set_xticklabels([f"${s:,}" for s in s_vals], rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Sales Value ($)", fontsize=11)
ax.set_ylabel("Predicted Profit ($)", fontsize=11)
ax.set_title("Scenario A: Profit from Sales", fontsize=12, fontweight="bold")
ax.set_facecolor("#F7F9FC"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for bar, val in zip(bars, p_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"${val:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
green_p = mpatches.Patch(color=COLORS[2], label="Profit"); red_p = mpatches.Patch(color=COLORS[3], label="Loss")
ax.legend(handles=[green_p, red_p], fontsize=9)

# Panel B: Profit from Discount
ax2 = axes[1]
d_vals = [d * 100 for d, _, _ in pred_b]
pb_vals= [p for _, p, _ in pred_b]
bar_c2 = [COLORS[2] if p >= 0 else COLORS[3] for p in pb_vals]
bars2  = ax2.bar(range(len(d_vals)), pb_vals, color=bar_c2, edgecolor="white", width=0.6)
ax2.axhline(0, color="black", linewidth=1.5)
ax2.axvline(disc_scenarios.index(min(disc_scenarios, key=lambda x: abs(x - breakeven))),
            color="orange", linewidth=2, linestyle="--",
            label=f"Break-even ≈{breakeven*100:.0f}%")
ax2.set_xticks(range(len(d_vals)))
ax2.set_xticklabels([f"{d:.0f}%" for d in d_vals], rotation=45, ha="right", fontsize=9)
ax2.set_xlabel("Discount Level", fontsize=11); ax2.set_ylabel("Predicted Profit ($)", fontsize=11)
ax2.set_title("Scenario B: Profit from Discount", fontsize=12, fontweight="bold")
ax2.set_facecolor("#F7F9FC"); ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=9)
for bar, val in zip(bars2, pb_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (2 if val >= 0 else -8),
             f"${val:.0f}", ha="center", fontsize=7, fontweight="bold")

# Panel C: Profit from Quantity
ax3 = axes[2]
q_vals  = [q for q, _ in pred_c]
pc_vals = [p for _, p in pred_c]
ax3.plot(q_vals, pc_vals, color=COLORS[2], linewidth=2.5, marker="o",
         markersize=7, markerfacecolor="white", markeredgewidth=2)
ax3.fill_between(q_vals, pc_vals, alpha=0.15, color=COLORS[2])
ax3.set_xlabel("Quantity Sold", fontsize=11); ax3.set_ylabel("Predicted Profit ($)", fontsize=11)
ax3.set_title("Scenario C: Profit from Quantity", fontsize=12, fontweight="bold")
ax3.set_facecolor("#F7F9FC"); ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
for q, p in pred_c:
    ax3.annotate(f"${p:.0f}", (q, p), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/fig10_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 10 saved → outputs/fig10_predictions.png")

# =============================================================================
#  STEP 5 — BUSINESS STRATEGY RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 65)
print("  STEP 5 — DATA-DRIVEN BUSINESS STRATEGY RECOMMENDATIONS")
print("=" * 65)

strategies = [
    {
        "title"  : "Strategy 1: Grow Sales Revenue as the Primary Profit Driver",
        "finding": f"Sales → Profit has the strongest relationship (r={m1['r']:.3f}, R²={m1['r2_test']:.3f}). "
                   f"Every $1 increase in Sales gives ${m1['coef']:.4f} more Profit.",
        "actions": [
            "Upsell higher-priced items (especially in Technology category).",
            "Bundle slow-moving products with best-sellers to increase basket value.",
            "Set revenue-based KPIs for sales staff rather than unit-count targets.",
            "Invest marketing budget in driving total revenue, not just transaction volume.",
        ],
    },
    {
        "title"  : "Strategy 2: Implement a Strict Discount Control Policy",
        "finding": f"Discount → Profit is negative (r={m2['r']:.3f}). "
                   f"Break-even discount = {breakeven*100:.1f}%. Discounts above this produce net losses on average.",
        "actions": [
            f"Cap standard discounts at 20% (predicted profit = ${predict(m2, 0.20):.2f}).",
            f"Prohibit discounts above {breakeven*100:.0f}% except for clearance sales.",
            "Replace blanket discounts with targeted loyalty discounts for high-value customers.",
            "Use time-limited flash sales (max 20%) instead of permanent price reductions.",
        ],
    },
    {
        "title"  : "Strategy 3: Product Bundling Over Individual Discounting",
        "finding": f"Quantity → Profit is weak (R²={m3['r2_test']:.4f}). "
                   "Selling more units alone does not drive proportional profit growth.",
        "actions": [
            "Bundle Office Supplies kits (e.g., stapler + binder clips + folders) at a fixed bundle price.",
            "Offer 'Buy 3, get a free item' instead of a per-unit discount — protects margin.",
            "Create category combo deals (Furniture + Office Supplies for new offices).",
        ],
    },
    {
        "title"  : "Strategy 4: Category-Level Profit Optimization",
        "finding": "Technology has the highest average sales and profit-per-transaction. "
                   "Furniture has high sales but lower profit ratios.",
        "actions": [
            "Technology: ensure stock availability — high sales = high profit.",
            "Furniture: review supplier contracts to improve cost margins.",
            "Office Supplies: optimize with bulk purchasing to reduce COGS.",
            "Track profit-per-category monthly and reallocate shelf/marketing space accordingly.",
        ],
    },
    {
        "title"  : "Strategy 5: Loyalty Program to Increase Transaction Frequency",
        "finding": "Repeat customers require lower marketing spend and tend to have higher basket values.",
        "actions": [
            "Launch a tiered loyalty program (Bronze/Silver/Gold) based on cumulative spend.",
            "Reward loyalty with exclusive product access or early-sale access — not discounts.",
            "Use RFM segmentation (Recency, Frequency, Monetary) to identify high-value customers.",
            "Target churning customers with personalised product recommendations.",
        ],
    },
]

for i, s in enumerate(strategies, 1):
    print(f"\n  {'─'*60}")
    print(f"  {s['title']}")
    print(f"  {'─'*60}")
    print(f"  Statistical Finding: {s['finding']}")
    print(f"  Recommended Actions:")
    for action in s["actions"]:
        print(f"    • {action}")

# =============================================================================
#  FIGURE 11 — Business Strategy Dashboard
# =============================================================================

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Figure 11 — Business Strategy Dashboard\n"
             "Data-Driven Retail Recommendations from Regression Analysis",
             fontsize=15, fontweight="bold")

# ── Panel 1: Profit margin by discount bracket (actual data)
ax1 = fig.add_subplot(2, 3, 1)
df["discount_bracket"] = pd.cut(df["Discount"],
    bins=[-0.01, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    labels=["0%", "1-10%", "11-20%", "21-30%", "31-40%", "41-50%", "51%+"])
bracket_profit = df.groupby("discount_bracket", observed=True)["Profit"].mean()
colors_bar = [COLORS[2] if v > 0 else COLORS[3] for v in bracket_profit]
bracket_profit.plot(kind="bar", ax=ax1, color=colors_bar, edgecolor="white", rot=30)
ax1.axhline(0, color="black", linewidth=1.5)
ax1.set_title("Avg Profit by Discount Bracket\n(Actual Data)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Discount Range"); ax1.set_ylabel("Average Profit ($)")
ax1.set_facecolor("#F7F9FC"); ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# ── Panel 2: Sales vs Profit by Category
ax2 = fig.add_subplot(2, 3, 2)
cat_colors = {"Furniture": COLORS[1], "Office Supplies": COLORS[0], "Technology": COLORS[2]}
for cat, grp in df.groupby("Category"):
    ax2.scatter(grp["Sales"], grp["Profit"], alpha=0.2, s=10,
                color=cat_colors.get(cat, "gray"), label=cat)
xline = np.linspace(df["Sales"].min(), df["Sales"].max(), 300)
ax2.plot(xline, m1["coef"] * xline + m1["intercept"],
         "k-", linewidth=2, label="Overall regression")
ax2.set_xlabel("Sales ($)"); ax2.set_ylabel("Profit ($)")
ax2.set_title("Sales vs Profit by Category", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8); ax2.set_facecolor("#F7F9FC")
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# ── Panel 3: Region profit comparison
ax3 = fig.add_subplot(2, 3, 3)
region_profit = df.groupby("Region")["Profit"].mean().sort_values(ascending=False)
region_profit.plot(kind="bar", ax=ax3, color=COLORS[:len(region_profit)],
                   edgecolor="white", rot=0)
ax3.set_title("Average Profit by Region", fontsize=11, fontweight="bold")
ax3.set_xlabel("Region"); ax3.set_ylabel("Average Profit ($)")
ax3.set_facecolor("#F7F9FC"); ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
for bar in ax3.patches:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"${bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")

# ── Panel 4: Profit per Category (boxplot)
ax4 = fig.add_subplot(2, 3, 4)
cat_list = list(df["Category"].unique())
data_by_cat = [df[df["Category"] == c]["Profit"].values for c in cat_list]
bp = ax4.boxplot(data_by_cat, patch_artist=True,
                 boxprops=dict(alpha=0.6),
                 medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
ax4.set_xticklabels(cat_list, fontsize=10)
ax4.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax4.set_title("Profit Distribution by Category", fontsize=11, fontweight="bold")
ax4.set_ylabel("Profit ($)"); ax4.set_facecolor("#F7F9FC")
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

# ── Panel 5: Predicted profit under different strategies
ax5 = fig.add_subplot(2, 3, 5)
strategy_labels = ["Zero\nDiscount", "10%\nDiscount", "20%\nDiscount",
                   "30%\nDiscount", "40%\nDiscount"]
strategy_preds  = [predict(m2, d) for d in [0.0, 0.1, 0.2, 0.3, 0.4]]
s_colors = [COLORS[2] if p > 0 else COLORS[3] for p in strategy_preds]
bars5 = ax5.bar(strategy_labels, strategy_preds, color=s_colors, edgecolor="white", width=0.5)
ax5.axhline(0, color="black", linewidth=1.5)
ax5.set_title("Predicted Profit at Discount Levels\n(Strategy Recommendation)",
              fontsize=11, fontweight="bold")
ax5.set_ylabel("Predicted Profit ($)"); ax5.set_facecolor("#F7F9FC")
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
for bar, val in zip(bars5, strategy_preds):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"${val:.1f}", ha="center", fontsize=9, fontweight="bold")

# ── Panel 6: Top 10 Sub-Categories by average Profit
ax6 = fig.add_subplot(2, 3, 6)
sub_profit = df.groupby("Sub-Category")["Profit"].mean().sort_values(ascending=True)
colors_sub = [COLORS[2] if v > 0 else COLORS[3] for v in sub_profit]
sub_profit.plot(kind="barh", ax=ax6, color=colors_sub, edgecolor="white")
ax6.axvline(0, color="black", linewidth=1.5)
ax6.set_title("Average Profit by Sub-Category", fontsize=11, fontweight="bold")
ax6.set_xlabel("Average Profit ($)"); ax6.set_facecolor("#F7F9FC")
ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/fig11_business_strategy_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 11 saved → outputs/fig11_business_strategy_dashboard.png")

# =============================================================================
#  Model Limitations
# =============================================================================

print("\n" + "=" * 65)
print("  MODEL LIMITATIONS")
print("=" * 65)
limitations = [
    "Low R² values (0.004–0.230): Each model uses only ONE predictor variable. "
    "In reality, profit depends on many factors simultaneously. "
    "Multiple Linear Regression would yield significantly better predictions.",
    "Outliers: Sales (11.7%) and Profit (18.8%) contain large outlier percentages "
    "that widen variance and pull regression lines.",
    "Linearity assumption: The true relationship may not be strictly linear "
    "(e.g., diminishing returns on advertising).",
    "Dataset lacks Advertising Cost and Store Area columns which the assignment "
    "specification lists as desirable — additional features would improve all models.",
    "No interaction terms: The combined effect of Discount AND Quantity on Profit "
    "is not captured by simple linear regression.",
    "Temporal effects ignored: Sales seasonality (holiday peaks, etc.) is not "
    "modelled. A time-series model would be more appropriate for forecasting.",
]
for i, lim in enumerate(limitations, 1):
    print(f"\n  {i}. {lim}")

print("\n" + "=" * 65)
print("  MEMBER 4 COMPLETE ✅")
print("=" * 65)
