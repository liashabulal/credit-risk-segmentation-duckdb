# Credit Risk Segmentation Using Synthetic Financial Behavior Data

A two-stage credit risk scoring system built on 500K customers and 5M transactions.

**Stage 1:** DuckDB SQL computes utilization rates and missed payment frequencies, 
producing a rules-based risk tier scorecard.

**Stage 2:** Logistic Regression and Random Forest models (CV AUC 0.95) improve 
on the SQL baseline using behavioral and demographic features.

**Key concepts demonstrated:**
- Synthetic data engineering with realistic financial relationships
- DuckDB for large scale SQL analytics
- Data leakage prevention
- Model comparison with cross validation

[View full notebook on Kaggle] : (https://www.kaggle.com/code/liashah1/credit-risk-segmentation-duckdb)
