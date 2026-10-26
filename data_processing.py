"""Data loading, validation and cleaning.

The generation, cleaning and feature-engineering logic below is reused as-is
from the midterm laboratory examination (Parts A and B).
"""
import numpy as np
import pandas as pd

TARGET = "Academic_Status"
ID_COL = "Student_ID"

BASE_FEATURES = [
    "Attendance_Rate",
    "Quiz_Average",
    "Assignment_Average",
    "Absences",
    "Late_Submissions",
    "Study_Hours",
    "Previous_GPA",
]
ENGINEERED_FEATURES = ["Engagement_Score", "Submission_Burden"]
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES  # exact midterm feature order

VALID_RANGES = {
    "Attendance_Rate": (0, 100),
    "Quiz_Average": (0, 100),
    "Assignment_Average": (0, 100),
    "Absences": (0, 100),
    "Late_Submissions": (0, 100),
    "Study_Hours": (0, 168),
    "Previous_GPA": (1, 5),
}

FIELD_DESCRIPTIONS = {
    "Student_ID": ("Unique student identifier", "Identifier"),
    "Attendance_Rate": ("Attendance percentage", "Predictor"),
    "Quiz_Average": ("Average quiz score", "Predictor"),
    "Assignment_Average": ("Average assignment score", "Predictor"),
    "Absences": ("Number of absences", "Predictor"),
    "Late_Submissions": ("Number of late submissions", "Predictor"),
    "Study_Hours": ("Weekly study hours", "Predictor"),
    "Previous_GPA": ("Previous GPA (1.0 = best, 5.0 = worst)", "Predictor"),
    "Academic_Status": ("Academic risk category", "Target"),
}


def generate_midterm_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Midterm Part A: fictional (synthetic) dataset, identical seed and logic."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, n + 1)],
        "Attendance_Rate": np.random.uniform(45, 100, n).round(1),
        "Quiz_Average": np.random.uniform(40, 100, n).round(1),
        "Assignment_Average": np.random.uniform(40, 100, n).round(1),
        "Absences": np.random.randint(0, 21, n),
        "Late_Submissions": np.random.randint(0, 11, n),
        "Study_Hours": np.random.uniform(1, 30, n).round(1),
        "Previous_GPA": np.random.uniform(1.0, 5.0, n).round(2),
    })
    risk_score = (
        (100 - df["Attendance_Rate"]) * 0.25
        + (100 - df["Quiz_Average"]) * 0.25
        + (100 - df["Assignment_Average"]) * 0.20
        + df["Absences"] * 0.80
        + df["Late_Submissions"] * 1.00
        + (30 - df["Study_Hours"]) * 0.50
        + (df["Previous_GPA"] - 1) * 5
    )
    df[TARGET] = np.where(risk_score > risk_score.quantile(0.60), "At Risk", "Satisfactory")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Midterm Part B feature engineering (same formulas and weights)."""
    out = df.copy()
    out["Engagement_Score"] = (
        out["Attendance_Rate"] * 0.40
        + out["Quiz_Average"] * 0.30
        + out["Assignment_Average"] * 0.30
    )
    out["Submission_Burden"] = out["Absences"] + out["Late_Submissions"]
    return out


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Midterm Part B cleaning: numeric coercion and range validation -> NaN."""
    out = df.copy()
    for col in BASE_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col, (low, high) in VALID_RANGES.items():
        out.loc[~out[col].between(low, high), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full midterm cleaning (labelled data): target tidy-up, de-duplication,
    label filtering, range validation and feature engineering."""
    out = df.copy()
    if TARGET in out.columns:
        out[TARGET] = out[TARGET].astype(str).str.strip().str.title()
    out = out.drop_duplicates().copy()
    if TARGET in out.columns:
        out = out[out[TARGET].isin(["At Risk", "Satisfactory"])].copy()
    out = clean_features(out)
    return add_engineered_features(out)


def validate_schema(df: pd.DataFrame):
    """Return (errors, warnings) for an uploaded dataframe."""
    errors, warnings = [], []
    if df is None or df.empty:
        return ["The file is empty: it has no data rows."], warnings
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
    if ID_COL not in df.columns:
        warnings.append(f"No '{ID_COL}' column found; row numbers will be used as IDs.")
    if TARGET not in df.columns:
        warnings.append(f"No '{TARGET}' column found; the file can be scored but model "
                        "accuracy against actual labels cannot be shown.")
    if not missing:
        for col in BASE_FEATURES:
            bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            if bad.any():
                warnings.append(f"'{col}': {int(bad.sum())} non-numeric value(s) will be treated as missing.")
    return errors, warnings


def quality_summary(df: pd.DataFrame) -> dict:
    """Null and duplicate summary for the Dataset page."""
    nulls = df.isnull().sum()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "total_nulls": int(nulls.sum()),
        "null_table": nulls.rename("Missing values").to_frame(),
    }


def count_out_of_range(df: pd.DataFrame) -> pd.Series:
    """Number of values per predictor outside the midterm valid ranges."""
    counts = {}
    for col, (low, high) in VALID_RANGES.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            counts[col] = int((vals.notna() & ~vals.between(low, high)).sum())
    return pd.Series(counts, name="Out-of-range values")


def prepare_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an uploaded file that already passed validate_schema()."""
    out = df.copy()
    if ID_COL not in out.columns:
        out.insert(0, ID_COL, [f"ROW{i + 1:03d}" for i in range(len(out))])
    return clean_dataset(out)
