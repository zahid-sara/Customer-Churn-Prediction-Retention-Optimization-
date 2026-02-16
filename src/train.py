"""End-to-end customer churn pipeline with EDA, modeling, SHAP, and ROI simulation."""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
FIG_DIR = Path("reports/figures")
MODEL_DIR = Path("model")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from Kaggle and place it in data/."
        )
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
        include_lowest=True,
    )
    out["monthly_to_tenure_ratio"] = out["MonthlyCharges"] / np.maximum(out["tenure"], 1)
    risk_map = {"Month-to-month": "High", "One year": "Medium", "Two year": "Low"}
    out["contract_risk_level"] = out["Contract"].map(risk_map)
    return out


def run_eda(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    corr_df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_df, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(df, x="MonthlyCharges", hue="Churn", bins=30, kde=True)
    plt.title("Monthly Charges Distribution by Churn")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "monthlycharges_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    churn_rate = df["Churn"].value_counts(normalize=True).mul(100)
    sns.barplot(x=churn_rate.index, y=churn_rate.values)
    plt.ylabel("Percent")
    plt.title("Churn Rate Analysis")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "churn_rate.png", dpi=200)
    plt.close()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def simulate_roi(df_scored: pd.DataFrame) -> dict[str, float]:
    df_sorted = df_scored.sort_values("churn_probability", ascending=False)
    top_n = int(0.2 * len(df_sorted))
    target = df_sorted.head(top_n).copy()

    incentive_cost = 20
    avg_revenue = 70

    retained_customers = target["Churn"].sum()
    revenue_saved = retained_customers * avg_revenue
    total_campaign_cost = len(target) * incentive_cost
    net_profit = revenue_saved - total_campaign_cost
    roi = net_profit / total_campaign_cost if total_campaign_cost > 0 else 0

    return {
        "customers_targeted": int(len(target)),
        "likely_churners_targeted": int(retained_customers),
        "revenue_saved": float(revenue_saved),
        "campaign_cost": float(total_campaign_cost),
        "net_profit": float(net_profit),
        "roi": float(roi),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    run_eda(df)
    df = add_features(df)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_transformed, y_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_transformed)
        y_proba = model.predict_proba(X_test_transformed)[:, 1]
        results[name] = evaluate(y_test, y_pred, y_proba)
        trained[name] = model

    results_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    results_df.to_csv("reports/model_comparison.csv", index=True)

    best_name = results_df.index[0]
    best_model = trained[best_name]

    y_best_proba = best_model.predict_proba(X_test_transformed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_best_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"Best model: {best_name}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=200)
    plt.close()

    best_pred = (y_best_proba >= 0.5).astype(int)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, best_pred, cmap="Blues")
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    if best_name == "xgboost":
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_transformed)

        plt.figure()
        shap.summary_plot(shap_values, X_test_transformed, show=False)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_summary.png", dpi=200)
        plt.close()

        sample_idx = 0
        force = shap.force_plot(
            explainer.expected_value,
            shap_values[sample_idx],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "shap_force_single_customer.png", dpi=200)
        plt.close()

    test_scored = X_test.copy()
    test_scored["Churn"] = y_test.values
    test_scored["churn_probability"] = y_best_proba
    roi = simulate_roi(test_scored)

    with open("reports/roi_summary.json", "w", encoding="utf-8") as f:
        json.dump(roi, f, indent=2)

    bundle = {
        "preprocessor": preprocessor,
        "model": best_model,
        "best_model_name": best_name,
        "metrics": results,
    }
    joblib.dump(bundle, MODEL_DIR / "churn_model.pkl")

    print("Training complete.")
    print(f"Best model: {best_name}")
    print("Artifacts saved in model/ and reports/")


if __name__ == "__main__":
    main()
