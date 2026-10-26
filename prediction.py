"""Model loading and inference. The saved pipeline is never retrained here."""
import json
import os
from datetime import datetime

import joblib
import pandas as pd

from modules.data_processing import (BASE_FEATURES, FEATURES, VALID_RANGES,
                                     add_engineered_features)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

LIMITATIONS = [
    "Trained on a fictional (synthetic) dataset of 300 students; it has not been validated on real student records.",
    "The 'At Risk' label in the training data was produced by a rule-based score, so the model mostly re-learns that rule rather than real-world academic outcomes.",
    "Recall for 'At Risk' is about 71% on the midterm test split: roughly 3 in 10 truly at-risk students would be missed.",
    "Probabilities are model estimates, not guarantees, and are not calibrated.",
    "A prediction is decision support only. It must never be used alone to deny services or penalize a student; a human reviewer records the final decision.",
]


def load_model():
    """Load the saved preprocessing + classifier pipeline (wrapped by st.cache_resource in app.py)."""
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


def input_errors(values: dict) -> list:
    """Validate a single-student input dict against the midterm valid ranges."""
    errors = []
    for col, (low, high) in VALID_RANGES.items():
        v = values.get(col)
        if v is None or pd.isna(v):
            errors.append(f"{col} is required.")
        elif not (low <= v <= high):
            errors.append(f"{col} must be between {low} and {high} (got {v}).")
    return errors


def build_input_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact training feature names and order."""
    row = pd.DataFrame([{c: values[c] for c in BASE_FEATURES}])
    return add_engineered_features(row)[FEATURES]


def _risk_index(model) -> int:
    return list(model.classes_).index("At Risk")


def predict_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score a dataframe that already contains all FEATURES. Returns label + probability."""
    X = df[FEATURES]
    out = pd.DataFrame(index=df.index)
    out["Predicted_Status"] = model.predict(X)
    out["At_Risk_Probability"] = model.predict_proba(X)[:, _risk_index(model)].round(3)
    return out


def predict_one(model, values: dict) -> dict:
    X = build_input_frame(values)
    res = predict_frame(model, X).iloc[0]
    return {
        "label": res["Predicted_Status"],
        "probability": float(res["At_Risk_Probability"]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": X,
    }


def risk_band(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"


def recommended_followup(p: float) -> str:
    band = risk_band(p)
    if band == "High":
        return "Priority review: adviser meets the student within a week; consider tutoring or counseling referral."
    if band == "Medium":
        return "Monitor: instructor checks attendance and submissions again in two weeks."
    return "No action needed beyond routine monitoring."
