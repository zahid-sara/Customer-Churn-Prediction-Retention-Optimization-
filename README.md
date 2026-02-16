# Customer Churn Prediction & Retention Optimization

End-to-end ML project to predict telecom customer churn, explain predictions with SHAP, and simulate retention campaign ROI.

## Problem Statement
Telecom providers lose recurring revenue when customers churn. This project predicts churn risk at customer level and turns predictions into a business action plan using an ROI simulation.

## Dataset
- **Primary source (Kaggle):** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- File expected in this repo as:
  - `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Repository Structure

```text
Customer-Churn-Prediction-Retention-Optimization-/
├── data/
├── notebooks/
│   └── churn_end_to_end.ipynb
├── src/
│   ├── train.py
│   └── inference.py
├── model/
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   └── roi_summary.json
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training Pipeline

```bash
python src/train.py
```

Pipeline includes:
1. Data understanding + EDA (missing values, imbalance, statistics, plots).
2. Data cleaning + feature engineering (`tenure_bucket`, `monthly_to_tenure_ratio`, `contract_risk_level`).
3. Modeling (Logistic Regression, Random Forest, XGBoost).
4. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve).
5. SHAP explainability (global summary + single-customer force plot).
6. Business ROI simulation for targeting top 20% high-risk customers with a $20 incentive.

## Model Comparison
Output table is saved to:
- `reports/model_comparison.csv`

The best model is auto-selected by ROC-AUC and serialized to:
- `model/churn_model.pkl`

## SHAP Explainability
Generated artifacts:
- `reports/figures/shap_summary.png`
- `reports/figures/shap_force_single_customer.png`

Business question answered:
- **Why did this customer churn?** Use the force plot for customer-level local explanation.

## Business ROI Simulation
Assumptions:
- Target top 20% highest-risk customers.
- Offer retention incentive = **$20/customer**.
- Average monthly revenue = **$70/customer**.

Saved output:
- `reports/roi_summary.json`

Formula used:

```text
Revenue Saved = (# likely churners targeted) × average revenue
Campaign Cost = (# targeted customers) × incentive
Net Profit = Revenue Saved - Campaign Cost
ROI = Net Profit / Campaign Cost
```

## Notebook
Use `notebooks/churn_end_to_end.ipynb` for a resume-ready walkthrough with markdown insights and visual analysis.

## Resume Entry (Sample)
**Customer Churn Prediction & Retention Optimization**
- Built end-to-end churn prediction pipeline using Logistic Regression, Random Forest, and XGBoost.
- Improved stakeholder trust through SHAP-based global and customer-level explainability.
- Designed retention ROI simulator to estimate campaign profitability before launch.
