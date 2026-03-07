# =============================================================================
#  MEMBER 3  —  Step 3: Regression Modeling  (Lines of Best Fit)
#  Assignment 1 | Linear Regression for Retail Business Analytics | IIUI FCIT
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("Sample_-_Superstore.csv", encoding="latin1")
COLORS = ["#2E75B6", "#E07B30", "#4CAF50", "#E53935", "#9C27B0", "#00ACC1"]

# =============================================================================
#  HELPER — fit one regression model with full metrics
# =============================================================================

def fit_model(xcol, ycol, data=df, test_size=0.20, seed=42):
    """
    Fit OLS Linear Regression.
    Returns a dictionary with the model, coefficients, and all metrics.
    """
    X = data[xcol].values.reshape(-1, 1)
    y = data[ycol].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    model = LinearRegression()
    model.fit(X_train, y_train)

    yp_train = model.predict(X_train)
    yp_test  = model.predict(X_test)

    return {
        "model"    : model,
        "xcol"     : xcol,
        "ycol"     : ycol,
        "coef"     : model.coef_[0],
        "intercept": model.intercept_,
        "X_train"  : X_train, "X_test" : X_test,
        "y_train"  : y_train, "y_test" : y_test,
        "yp_train" : yp_train,"yp_test": yp_test,
        "r2_train" : r2_score(y_train, yp_train),
        "r2_test"  : r2_score(y_test,  yp_test),
        "mse"      : mean_squared_error(y_test,  yp_test),
        "rmse"     : np.sqrt(mean_squared_error(y_test, yp_test)),
        "mae"      : mean_absolute_error(y_test,  yp_test),
        "r"        : np.corrcoef(data[xcol], data[ycol])[0, 1],
    }

def print_model_results(res, model_num):
    xcol, ycol = res["xcol"], res["ycol"]
    c, b       = res["coef"], res["intercept"]
    sign       = "+" if b >= 0 else "-"
    print(f"\n  {'─'*60}")
    print(f"  MODEL {model_num}:  {ycol}  =  f( {xcol} )")
    print(f"  {'─'*60}")
    print(f"  Regression Equation : {ycol} = {c:.4f} × {xcol} {sign} {abs(b):.4f}")
    print(f"  Slope (β₁)          : {c:.4f}")
    if c > 0:
        print(f"  Interpretation      : Every 1-unit ↑ in {xcol} → {ycol} ↑ by {c:.4f}")
    else:
        print(f"  Interpretation      : Every 1-unit ↑ in {xcol} → {ycol} ↓ by {abs(c):.4f}")
    print(f"  Intercept (β₀)      : {b:.4f}")
    print(f"  Pearson r           : {res['r']:.4f}")
    print(f"  R² (train)          : {res['r2_train']:.4f}  →  explains {res['r2_train']*100:.1f}% of {ycol} variance")
    print(f"  R² (test)           : {res['r2_test']:.4f}  →  explains {res['r2_test']*100:.1f}% of {ycol} variance")
    print(f"  RMSE (test)         : {res['rmse']:.4f}")
    print(f"  MAE  (test)         : {res['mae']:.4f}")
    print(f"  MSE  (test)         : {res['mse']:.4f}")

# =============================================================================
#  FIT ALL 4 MODELS
# =============================================================================

print("=" * 65)
print("  STEP 3 — LINEAR REGRESSION MODELS (Lines of Best Fit)")
print("=" * 65)
print("  Data split: 80% training | 20% testing | random_state=42")

m1 = fit_model("Sales",    "Profit")   # Strongest relationship
m2 = fit_model("Discount", "Profit")   # Key risk model
m3 = fit_model("Quantity", "Profit")   # Weak model
m4 = fit_model("Discount", "Sales")    # Negligible model

print_model_results(m1, 1)
print_model_results(m2, 2)
print_model_results(m3, 3)
print_model_results(m4, 4)

# Break-even discount
be_discount = -m2["intercept"] / m2["coef"]
print(f"\n  ⚠  CRITICAL FINDING (Model 2):")
print(f"     Break-even discount = {be_discount*100:.1f}%")
print(f"     Discounts above {be_discount*100:.1f}% are predicted to produce a LOSS on average.")

# =============================================================================
#  SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 95)
print("  MODEL COMPARISON SUMMARY")
print("=" * 95)
print(f"  {'#':<4} {'X':<12} {'Y':<10} {'Equation':<42}"
      f" {'r':>7} {'R²(test)':>9} {'RMSE':>9} {'MAE':>8}")
print("  " + "-" * 95)
for i, m in enumerate([m1, m2, m3, m4], 1):
    sign = "+" if m["intercept"] >= 0 else "-"
    eq   = f"{m['ycol']} = {m['coef']:.4f}×{m['xcol']} {sign} {abs(m['intercept']):.2f}"
    print(f"  {i:<4} {m['xcol']:<12} {m['ycol']:<10} {eq:<42}"
          f" {m['r']:>7.4f} {m['r2_test']:>9.4f} {m['rmse']:>9.4f} {m['mae']:>8.4f}")

print(f"\n  Best predictive model : Model 1 (Sales → Profit, highest R²)")
print(f"  Most dangerous finding: Model 2 (Discount → Profit, negative slope)")

# =============================================================================
#  FIGURE 5 — Model 1: Sales → Profit (3-panel)
# =============================================================================

def plot_three_panel(res, fig_num, color, filename):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    xcol, ycol = res["xcol"], res["ycol"]
    fig.suptitle(f"Figure {fig_num} — Model: {ycol}  =  f({xcol})",
                 fontsize=14, fontweight="bold")

    # Panel A: Scatter + regression line
    ax = axes[0]
    ax.scatter(res["X_train"].flatten(), res["y_train"],
               alpha=0.12, s=10, color=color, label="Train set")
    ax.scatter(res["X_test"].flatten(),  res["y_test"],
               alpha=0.35, s=14, color="#555555", label="Test set")
    xline = np.linspace(res["X_train"].min(), res["X_train"].max(), 300)
    yline = res["coef"] * xline + res["intercept"]
    ax.plot(xline, yline, "k-", linewidth=2.5,
            label=f"y={res['coef']:.4f}x+{res['intercept']:.2f}")
    ax.set_xlabel(xcol, fontsize=11); ax.set_ylabel(ycol, fontsize=11)
    ax.set_title(f"Regression Line  (R²={res['r2_test']:.4f})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_facecolor("#F7F9FC")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel B: Actual vs Predicted
    ax2 = axes[1]
    ax2.scatter(res["y_test"], res["yp_test"],
                alpha=0.25, s=12, color=color)
    lims = [min(res["y_test"].min(), res["yp_test"].min()) * 1.05,
            max(res["y_test"].max(), res["yp_test"].max()) * 1.05]
    ax2.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction (45°)")
    ax2.set_xlabel(f"Actual {ycol}", fontsize=11)
    ax2.set_ylabel(f"Predicted {ycol}", fontsize=11)
    ax2.set_title("Actual vs Predicted", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_facecolor("#F7F9FC")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # Panel C: Residual plot
    ax3 = axes[2]
    residuals = res["y_test"] - res["yp_test"]
    ax3.scatter(res["yp_test"], residuals,
                alpha=0.25, s=12, color="#E53935")
    ax3.axhline(0, color="black", linewidth=2, linestyle="--")
    ax3.set_xlabel(f"Predicted {ycol}", fontsize=11)
    ax3.set_ylabel("Residuals  (Actual − Predicted)", fontsize=11)
    ax3.set_title("Residual Plot", fontsize=12)
    ax3.set_facecolor("#F7F9FC")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # Residual stats annotation
    rmse_txt = (f"RMSE = {res['rmse']:.2f}\n"
                f"MAE  = {res['mae']:.2f}\n"
                f"Mean residual = {residuals.mean():.2f}")
    ax3.text(0.97, 0.97, rmse_txt, transform=ax3.transAxes, fontsize=9,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.9, edgecolor="gray"))

    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  ✅ Figure {fig_num} saved → outputs/{filename}")

plot_three_panel(m1, 5, COLORS[1], "fig5_model1_sales_profit.png")
plot_three_panel(m2, 6, COLORS[3], "fig6_model2_discount_profit.png")
plot_three_panel(m3, 7, COLORS[2], "fig7_model3_quantity_profit.png")

# =============================================================================
#  FIGURE 8 — All 4 Models side-by-side (regression lines)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Figure 8 — All Regression Models: Lines of Best Fit",
             fontsize=15, fontweight="bold")

model_meta = [
    (m1, COLORS[1], "Model 1: Sales → Profit  (Strongest)"),
    (m2, COLORS[3], "Model 2: Discount → Profit  (Key Risk)"),
    (m3, COLORS[2], "Model 3: Quantity → Profit  (Weak)"),
    (m4, COLORS[0], "Model 4: Discount → Sales   (Negligible)"),
]

for ax, (res, color, title) in zip(axes.flat, model_meta):
    xcol, ycol = res["xcol"], res["ycol"]
    ax.scatter(df[xcol], df[ycol], alpha=0.10, s=8, color=color)
    xline = np.linspace(df[xcol].min(), df[xcol].max(), 300)
    yline = res["coef"] * xline + res["intercept"]
    ax.plot(xline, yline, "k-", linewidth=2.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xcol, fontsize=11); ax.set_ylabel(ycol, fontsize=11)

    sign = "+" if res["intercept"] >= 0 else "−"
    eq   = f"{ycol} = {res['coef']:.4f}×{xcol} {sign} {abs(res['intercept']):.2f}"
    box_txt = (f"{eq}\n"
               f"Pearson r = {res['r']:.4f}\n"
               f"R² (test) = {res['r2_test']:.4f}\n"
               f"RMSE      = {res['rmse']:.2f}")
    ax.text(0.97, 0.97, box_txt, transform=ax.transAxes, fontsize=10,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      alpha=0.88, edgecolor="gray"))
    ax.set_facecolor("#F7F9FC")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/fig8_all_models_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 8 saved → outputs/fig8_all_models_comparison.png")

# =============================================================================
#  FIGURE 9 — Residual Histograms for all 4 models
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Figure 9 — Residual Distributions (All Models)",
             fontsize=13, fontweight="bold")

for ax, (res, color, title) in zip(axes, model_meta):
    residuals = res["y_test"] - res["yp_test"]
    ax.hist(residuals, bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=2, linestyle="--")
    ax.axvline(residuals.mean(), color="red", linewidth=1.5, linestyle=":",
               label=f"Mean={residuals.mean():.1f}")
    ax.set_title(f"{res['ycol']} vs {res['xcol']}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F7F9FC")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/fig9_residual_histograms.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✅ Figure 9 saved → outputs/fig9_residual_histograms.png")

# Export model results for use in step 4
import pickle
with open("outputs/models.pkl", "wb") as f:
    pickle.dump({"m1": m1, "m2": m2, "m3": m3, "m4": m4}, f)
print("\n  ✅ Model objects saved → outputs/models.pkl (used by step4)")

print("\n" + "=" * 65)
print("  MEMBER 3 COMPLETE ✅")
print("=" * 65)
