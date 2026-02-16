"""Inference utility for churn model."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("model/churn_model.pkl")


def predict_churn(customer_df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    X = preprocessor.transform(customer_df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = customer_df.copy()
    out["churn_probability"] = probs
    out["churn_prediction"] = preds
    return out
