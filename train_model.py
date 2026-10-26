"""Run ONCE offline to rebuild the saved model from the midterm pipeline.

This reuses the midterm code (data generation, cleaning, split, preprocessing,
Decision Tree vs Random Forest comparison). The Streamlit app never retrains:
it only loads models/trained_model.joblib.

    python train_model.py
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from modules.data_processing import (BASE_FEATURES, FEATURES, TARGET,
                                     add_engineered_features,
                                     clean_dataset, generate_midterm_dataset)

MODEL_VERSION = "1.0.0-midterm-rf"

# ---- Midterm Part A / B ------------------------------------------------
raw = generate_midterm_dataset()
raw.to_csv("data/student_performance.csv", index=False)
df = clean_dataset(raw)
df.to_csv("data/student_performance_cleaned.csv", index=False)

# ---- Midterm Part C ----------------------------------------------------
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

preprocessor = ColumnTransformer([("numeric", SimpleImputer(strategy="median"), FEATURES)])
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,
                                            class_weight="balanced"),
}

labels = ["At Risk", "Satisfactory"]
metrics, matrices, pipelines = {}, {}, {}
for name, model in models.items():
    pipe = Pipeline([("preprocessing", preprocessor), ("classifier", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    pipelines[name] = pipe
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (At Risk)": precision_score(y_test, pred, pos_label="At Risk", zero_division=0),
        "Recall (At Risk)": recall_score(y_test, pred, pos_label="At Risk", zero_division=0),
        "F1 (At Risk)": f1_score(y_test, pred, pos_label="At Risk", zero_division=0),
    }
    matrices[name] = confusion_matrix(y_test, pred, labels=labels).tolist()

chosen = pipelines["Random Forest"]  # same model chosen in the midterm
rf = chosen.named_steps["classifier"]
importance = (pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_})
              .sort_values("Importance", ascending=False))

# ---- Five fictional new students (midterm) ----------------------------
new_students = pd.DataFrame([
    [50, 45, 50, 12, 6, 3, 3.8],
    [95, 90, 92, 1, 0, 20, 1.8],
    [65, 60, 62, 8, 4, 6, 3.2],
    [88, 82, 85, 2, 1, 15, 2.2],
    [72, 70, 68, 5, 3, 10, 2.8],
], columns=BASE_FEATURES)
new_students = add_engineered_features(new_students)
sample = new_students[BASE_FEATURES].copy()
sample.insert(0, "Student_ID", [f"NEW{i}" for i in range(1, 6)])
sample["Predicted_Status"] = chosen.predict(new_students[FEATURES])
proba = chosen.predict_proba(new_students[FEATURES])
sample["Estimated_At_Risk_Probability"] = proba[:, list(chosen.classes_).index("At Risk")].round(3)
sample.to_csv("data/five_sample_predictions.csv", index=False)

# ---- Save artefacts ----------------------------------------------------
joblib.dump(chosen, "models/trained_model.joblib")
metadata = {
    "model_name": "Random Forest (100 trees, class_weight=balanced)",
    "model_version": MODEL_VERSION,
    "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "sklearn_version": sklearn.__version__,
    "target": TARGET,
    "features": FEATURES,
    "classes": list(chosen.classes_),
    "n_rows_after_cleaning": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "metrics": metrics,
    "confusion_matrices": matrices,
    "confusion_labels": labels,
    "feature_importance": importance.to_dict(orient="records"),
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
pd.DataFrame(metrics).T.to_csv("data/model_evaluation.csv")

print(pd.DataFrame(metrics).T.round(3).to_string())
print(importance.round(3).to_string(index=False))
print(sample.to_string(index=False))
print("Saved models/trained_model.joblib")
