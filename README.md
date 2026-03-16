# Retail Business Analytics — Data Science Pipeline

A complete **data science pipeline** applied to the **Superstore retail dataset**, covering:

- Statistical **Exploratory Data Analysis (EDA)**
- **Correlation analysis**
- **OLS regression modeling**
- **Predictive scenario simulation**
- **Data-driven business strategy recommendations**

The project analyzes **9,994 retail transactions** using **8 numerical variables** to uncover actionable business insights.

---

## Key Results

| Relationship | r | R² | Business Finding |
|---|---|---|---|
| Advertising Cost → Sales | 0.775 | 0.564 | $1 ad spend generates **$5.93 in sales** |
| Sales → Profit | 0.479 | — | **$0.258 profit per $1 revenue** |
| Discount → Profit | -0.220 | 0.064 | **Break-even at 28.4% discount** |
| Customer Visits → Spending | 0.520 | 0.255 | Each visit adds **$28.06 spending** |
| Quantity → Profit | 0.066 | 0.002 | Volume alone **does not drive profit** |
| Discount → Sales | -0.028 | ≈0 | Discounting **does not increase volume** |

---

## Project Structure

retail-business-analytics/
│
├── main.py
│ Master runner — executes all pipeline steps in sequence
│
├── step1_eda_statistics.py
│ Performs descriptive statistics and IQR-based outlier detection
│
├── step2_scatter_correlation.py
│ Generates scatter plots and Pearson correlation heatmap
│
├── step3_regression_models.py
│ Builds OLS regression models and performs residual diagnostics
│
├── step4_predictions_strategy.py
│ Runs prediction scenarios and derives business strategies
│
├── Sample_Superstore.csv
│ Superstore dataset — 9,994 rows and 25 columns
│
└── outputs/
Auto-generated visualizations and analysis results
│
├── fig1_distributions_boxplots.png
├── fig2_scatter_plots.png
├── fig3_correlation.png
├── fig4_pairplot.png
├── fig5_all_models.png
├── fig6_model1_adcost_sales.png
├── fig7_model2_sales_profit.png
├── fig8_model5_visits_spending.png
├── fig9_residual_histograms.png
├── fig10_predictions.png
└── fig11_business_strategy_dashboard.png

---

## Dataset

**Base Dataset**

Superstore Sales Dataset (Kaggle)

- **9,994 records**
- **21 original columns**
- Time range: **2014–2017**

### Engineered Features

Four synthetic variables were generated to simulate realistic business drivers.

| Column | Correlated With | r |
|---|---|---|
| Advertising_Cost | Sales (category-weighted) | 0.775 |
| Store_Area | Sales (region-stratified) | 0.488 |
| Customer_Visits | Quantity (segment-stratified) | 0.787 |
| Customer_Spending | Sales (visit-adjusted) | 0.988 |

Synthetic variables were generated using **business-logical relationships with Gaussian noise** to maintain realistic statistical distributions.

---

## Installation

Install required dependencies:
```bash pip install pandas numpy matplotlib seaborn scikit-learn```

Run the full pipeline:
```python main.py``` 

All 11 figures will automatically save in the outputs/ directory.

--- 

## Regression Models

| Model | Predictor → Target | Equation | R² | RMSE |
|------|-------------------|----------|----|------|
| M1 | Advertising Cost → Sales | Sales = 5.93 × AdCost − 324.38 | 0.564 | 507.5 |
| M2 | Sales → Profit | Profit = 0.258 × Sales − 27.44 | — | 295.0 |
| M3 | Discount → Profit | Profit = −241.97 × Discount + 68.63 | 0.064 | 213.1 |
| M4 | Quantity → Profit | Profit = 7.09 × Quantity + 3.98 | 0.002 | 219.9 |
| M5 | Visits → Spending | Spending = 28.06 × Visits − 247.63 | 0.255 | 635.6 |
| M6 | Discount → Sales | Sales = −126.75 × Discount + 245.51 | ≈0 | 769.9 |

---

## Predictive Scenarios

| Scenario | Input | Predicted Output |
|---------|------|------------------|
| Advertising Budget | $300/month | $1,455 in sales |
| Sales Transaction | $500 sale | $101.39 profit |
| Customer Visits | 30 visits | $594 spending |
| Discount Level | > 28.4% | Net loss (break-even threshold) |

---

## Business Strategies

| # | Strategy | Statistical Basis |
|---|----------|-------------------|
| 1 | Maintain ≥ $300/month advertising | Model M1 shows **5.93× revenue return** |
| 2 | Avoid transactions below $107 | Model M2 shows **average loss below this level** |
| 3 | Cap discounts at **20%** | Break-even occurs at **28.4% discount** |
| 4 | Increase customer visit frequency | Each visit adds **$28.06 spending** |
| 5 | Bundle products instead of discounting | Quantity explains only **0.2% of profit** |

--- 


## Technology Stack

| Category | Tools |
|--------|------|
| Programming Language | Python 3.10 |
| Data Processing | pandas, numpy |
| Data Visualization | matplotlib, seaborn |
| Machine Learning / Modeling | scikit-learn |
| Statistical Analysis | OLS Regression, Correlation Analysis |
| Environment | Jupyter / Python Scripts |
| Version Control | Git & GitHub |

--- 

