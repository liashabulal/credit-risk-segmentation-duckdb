# Credit Risk Segmentation

## Overview

This project builds a two-stage credit risk scoring pipeline on a synthetic dataset of 500,000 customers and 5,000,000 transactions.

- **Stage 1 (SQL scorecard):** DuckDB aggregates the transaction data to the customer level and computes two behavioral signals — utilization rate (average monthly spend relative to credit limit, capped at 100%) and missed payment percentage. These are combined into a `raw_risk_score` and used to sort customers into four risk tiers: Low, Medium, High, and Very High Risk.
- **Stage 2 (machine learning):** Logistic Regression and Random Forest models are trained on the SQL-derived features (utilization rate, months on book, credit limit, raw risk score) to see whether they can improve on the rules-based scorecard. `missed_pct` is deliberately excluded from the model features to avoid data leakage, since it is derived from the same process that defines the target.

The final output combines the SQL risk tier and the model's predicted probability into a per-customer decision: Approve, Review, or Decline.

## Key results

- 5-fold cross-validation AUC: **Logistic Regression ~0.951**, **Random Forest ~0.942**.
- Logistic Regression slightly outperforms Random Forest, suggesting the relationships between the behavioral features and risk are largely linear.
- Decision thresholds on the Logistic Regression probability: below 0.3 → Approve, 0.3–0.6 → Review, 0.6 and above → Decline. On the last full run this produced roughly 364K Approve, 94K Decline, and 42K Review out of 500,000 customers.
- In the Random Forest, the SQL-derived `raw_risk_score` was the strongest feature, followed by utilization rate, months on book, and credit limit — indicating the rules-based scorecard already captures a meaningful share of the signal.

## Tech stack

- Python 3.11
- pandas, numpy — data generation and manipulation
- duckdb — SQL aggregation over the transaction data
- scikit-learn — Logistic Regression, Random Forest, cross-validation, pipelines
- matplotlib, seaborn — dashboard charts
- uv — dependency installation into the project's virtual environment

## How to run

1. Activate the project's virtual environment (`.venv`).
2. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```
3. Open `credit-risk-segmentation-duckdb.ipynb` in Jupyter or VS Code.
4. Run all cells in order, top to bottom. The notebook generates the synthetic data, runs the DuckDB scorecard, trains and evaluates the models, and saves a dashboard image (`credit_risk_dashboard.png`) to the project directory.

## Optimizations applied

The notebook originally contained a few performance and maintainability issues, documented in detail in `NOTES.md`. The following fixes have been applied:

- **`credit_limit` generation:** replaced a Python loop that called `np.random.choice` once per customer (500,000 calls) with a vectorized version that splits customers into two risk groups and draws credit limits for each group in a single batched call.
- **`customer_risks` lookup:** replaced a dictionary built from 500,000 customers and looked up in a Python loop over 5,000,000 transactions with direct numpy array indexing, since `customer_id` runs consecutively from 1 to n.
- **Redundant model training:** each model (Logistic Regression, Random Forest) was previously being fit from scratch up to three times — once implicitly during cross-validation, once again for the feature-importance chart, and once more for the final predictions. Each model is now fit once and the fitted pipeline is reused everywhere it's needed. This also removed a duplicated `features` list that had been defined separately in two cells.
