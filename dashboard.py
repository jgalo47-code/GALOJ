"""Decision-support dashboard: KPIs, risk/review queue and human decision log."""
from datetime import datetime

import pandas as pd
import plotly.express as px

from modules.data_processing import BASE_FEATURES, ID_COL, TARGET
from modules.prediction import predict_frame, recommended_followup, risk_band

DECISIONS = ["Pending review", "No action needed", "Monitor", "Refer to adviser / tutoring",
             "Refer to counseling", "Data issue - recheck record"]


def score_students(model, df: pd.DataFrame) -> pd.DataFrame:
    """Attach prediction, probability, risk band and follow-up to each student."""
    scored = df.copy()
    pred = predict_frame(model, scored)
    scored["Predicted_Status"] = pred["Predicted_Status"].values
    scored["At_Risk_Probability"] = pred["At_Risk_Probability"].values
    scored["Risk_Band"] = scored["At_Risk_Probability"].map(risk_band)
    scored["Recommended_Follow_up"] = scored["At_Risk_Probability"].map(recommended_followup)
    return scored


def kpis(scored: pd.DataFrame, threshold: float) -> dict:
    flagged = scored[scored["At_Risk_Probability"] >= threshold]
    return {
        "Students scored": len(scored),
        "Flagged for review": len(flagged),
        "Flagged (%)": round(len(flagged) / len(scored) * 100, 1) if len(scored) else 0.0,
        "High-risk band": int((scored["Risk_Band"] == "High").sum()),
        "Avg at-risk probability": round(float(scored["At_Risk_Probability"].mean()), 3),
        "Avg attendance (flagged)": round(float(flagged["Attendance_Rate"].mean()), 1) if len(flagged) else 0.0,
    }


def review_queue(scored: pd.DataFrame, threshold: float, decisions: dict) -> pd.DataFrame:
    """Students at or above the threshold, highest probability first."""
    q = scored[scored["At_Risk_Probability"] >= threshold].copy()
    q["Review_Status"] = q[ID_COL].map(lambda s: decisions.get(s, {}).get("decision", "Pending review"))
    cols = [ID_COL, "At_Risk_Probability", "Risk_Band", "Review_Status", "Attendance_Rate",
            "Quiz_Average", "Assignment_Average", "Absences", "Late_Submissions",
            "Study_Hours", "Previous_GPA", "Recommended_Follow_up"]
    return q.sort_values("At_Risk_Probability", ascending=False)[cols]


def risk_band_chart(scored: pd.DataFrame):
    counts = (scored["Risk_Band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
              .rename_axis("Risk band").reset_index(name="Students"))
    return px.bar(counts, x="Risk band", y="Students", color="Risk band", text="Students",
                  color_discrete_map={"High": "#d1495b", "Medium": "#edae49", "Low": "#2e86ab"},
                  title="Students by risk band")


def probability_histogram(scored: pd.DataFrame):
    return px.histogram(scored, x="At_Risk_Probability", nbins=20,
                        title="Distribution of at-risk probability")


def record_decision(decisions: dict, student_id: str, reviewer: str, decision: str, notes: str):
    decisions[student_id] = {
        "decision": decision,
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def decisions_table(decisions: dict) -> pd.DataFrame:
    if not decisions:
        return pd.DataFrame(columns=[ID_COL, "decision", "reviewer", "notes", "recorded_at"])
    return (pd.DataFrame.from_dict(decisions, orient="index")
            .rename_axis(ID_COL).reset_index())
