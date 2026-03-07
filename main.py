# =============================================================================
#  main.py  — Run the complete assignment in sequence
#  Assignment 1 | Linear Regression for Retail Business Analytics | IIUI FCIT
# =============================================================================
#
#  HOW TO RUN IN VS CODE:
#  1. Open this folder in VS Code
#  2. pip install -r requirements.txt
#  3. Place Sample_-_Superstore.csv in this folder
#  4. Press F5  or right-click -> "Run Python File in Terminal"
#
# =============================================================================

import os, sys, time, importlib
os.makedirs("outputs", exist_ok=True)

print("""
╔══════════════════════════════════════════════════════════════╗
║   IIUI FCIT  |  Artificial Intelligence  |  Assignment 1    ║
║   Linear Regression for Retail Business Analytics           ║
║   Dataset: Sample Superstore (Kaggle)  |  9,994 records     ║
╚══════════════════════════════════════════════════════════════╝
""")

steps = [
    ("step1_eda_statistics",      "Member 1 — EDA & Descriptive Statistics"),
    ("step2_scatter_correlation", "Member 2 — Scatter Plots & Correlation"),
    ("step3_regression_models",   "Member 3 — Regression Modeling"),
    ("step4_predictions_strategy","Member 4 — Predictions & Business Strategy"),
]

for module, label in steps:
    print(f"\n{'='*65}")
    print(f"  RUNNING: {label}")
    print(f"{'='*65}\n")
    importlib.import_module(module)

print(f"\n{'='*65}")
print("  ALL STEPS COMPLETE")
print(f"{'='*65}")
print("\n  Output figures saved in outputs/ folder:")
for f in sorted(os.listdir("outputs")):
    if f.endswith(".png"):
        print(f"    -> {f}")
