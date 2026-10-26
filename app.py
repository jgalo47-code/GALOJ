"""Predictive Student Academic Risk Monitoring - Streamlit deployment of the midterm model."""
import os

import pandas as pd
import streamlit as st

from modules import dashboard, exploratory_analysis as eda, prediction
from modules.data_processing import (BASE_FEATURES, FEATURES, FIELD_DESCRIPTIONS, ID_COL,
                                     TARGET, VALID_RANGES, count_out_of_range,
                                     prepare_uploaded, quality_summary, validate_schema)

st.set_page_config(page_title="SAD Predictive System", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "data", "student_performance_cleaned.csv")
DEFAULT_RAW = os.path.join(BASE_DIR, "data", "student_performance.csv")
DIAGRAM = os.path.join(BASE_DIR, "assets", "use_case_diagram.png")


@st.cache_resource
def load_model():
    return prediction.load_model()


@st.cache_data
def load_metadata():
    return prediction.load_metadata()


@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA)


@st.cache_data
def load_default_raw():
    return pd.read_csv(DEFAULT_RAW)


@st.cache_data
def score_cached(df: pd.DataFrame):
    return dashboard.score_students(load_model(), df)


model = load_model()
meta = load_metadata()

# ---- session state ----------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = load_default_data()
    st.session_state.data_source = "Bundled synthetic midterm dataset (cleaned)"
if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []

st.sidebar.title("Academic Risk Monitor")
menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction",
                                       "Dashboard", "System Design"])
st.sidebar.caption(f"Model: {meta['model_version']}")
st.sidebar.caption(f"Data: {st.session_state.data_source}")
st.sidebar.warning("Decision support only. A human reviewer makes every final decision.")


def labelled(df):
    return TARGET in df.columns and df[TARGET].notna().any()


# =====================================================================
if menu == "Home":
    st.title("Predictive Student Academic Risk Monitoring")
    st.caption("Systems Analysis and Design - Laboratory Exercise: deploying the midterm data-mining output")
    st.markdown(
        "This app deploys the **Random Forest pipeline from the midterm examination**. It flags "
        "students whose attendance, quiz and assignment results, absences, late submissions, "
        "study hours and previous GPA suggest a risk of poor academic standing, so that an "
        "adviser or instructor can review them early."
    )
    c1, c2, c3, c4 = st.columns(4)
    m = meta["metrics"]["Random Forest"]
    c1.metric("Accuracy", f"{m['Accuracy']:.1%}")
    c2.metric("Precision (At Risk)", f"{m['Precision (At Risk)']:.1%}")
    c3.metric("Recall (At Risk)", f"{m['Recall (At Risk)']:.1%}")
    c4.metric("F1 (At Risk)", f"{m['F1 (At Risk)']:.1%}")
    st.subheader("Modules")
    st.markdown(
        "- **Dataset** - upload a CSV, check its schema, preview it, and see null and duplicate counts.\n"
        "- **Data Analysis** - descriptive statistics, class distribution and interactive charts.\n"
        "- **Prediction** - enter one student's values or score a batch; see label, probability and model metrics.\n"
        "- **Dashboard** - KPIs, risk/review queue, and recording of the human follow-up decision.\n"
        "- **System Design** - requirements, actors, use case diagram and decision workflow."
    )
    st.subheader("Continuity with the midterm")
    st.markdown(
        "- The project is **Predictive Student Academic Risk Monitoring**, the problem actually solved in the midterm. "
        "It differs from the facility-reservation topic listed for Section A in the lab sheet; the midterm "
        "target (`Academic_Status`), features, dataset and results are reused unchanged.\n"
        "- Data is **synthetic** (generated with seed 42), so it is safe for public deployment.\n"
        f"- Model: {meta['model_name']}, trained {meta['trained_at_utc']} (scikit-learn {meta['sklearn_version']})."
    )

# =====================================================================
elif menu == "Dataset":
    st.title("Dataset Upload and Validation")
    st.markdown("Upload a CSV to replace the bundled dataset. Required columns: "
                + ", ".join(f"`{c}`" for c in BASE_FEATURES)
                + f". Optional: `{ID_COL}`, `{TARGET}`. Use synthetic or authorized data only.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except pd.errors.EmptyDataError:
            st.error("The file is empty. Please upload a CSV with a header row and data.")
            raw = None
        except Exception as exc:  # malformed / non-CSV content
            st.error(f"Could not read the file as CSV: {exc}")
            raw = None
        if raw is not None:
            errors, warnings = validate_schema(raw)
            for w in warnings:
                st.warning(w)
            if errors:
                for e in errors:
                    st.error(e)
                st.info("Fix the file and upload it again. The previously loaded dataset is unchanged.")
            else:
                st.session_state.data = prepare_uploaded(raw)
                st.session_state.data_source = f"Uploaded: {uploaded.name}"
                st.success(f"{uploaded.name} passed validation and is now the active dataset.")
    if st.button("Reset to bundled midterm dataset"):
        st.session_state.data = load_default_data()
        st.session_state.data_source = "Bundled synthetic midterm dataset (cleaned)"
        st.rerun()

    df = st.session_state.data
    st.subheader("Active dataset")
    st.caption(st.session_state.data_source)
    st.dataframe(df.head(20), use_container_width=True)
    q = quality_summary(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", q["rows"])
    c2.metric("Columns", q["columns"])
    c3.metric("Duplicate rows", q["duplicate_rows"])
    c4.metric("Missing values", q["total_nulls"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values per column**")
        st.dataframe(q["null_table"], use_container_width=True)
    with right:
        st.markdown("**Out-of-range values (set to missing during cleaning)**")
        st.dataframe(count_out_of_range(df).to_frame(), use_container_width=True)
    st.markdown("**Data dictionary**")
    st.dataframe(pd.DataFrame([(k, v[0], v[1]) for k, v in FIELD_DESCRIPTIONS.items()],
                              columns=["Field", "Description", "Type"]),
                 use_container_width=True, hide_index=True)
    st.caption("Cleaning follows the midterm: non-numeric and out-of-range values become missing and are "
               "imputed with the training median inside the saved pipeline; duplicates are dropped.")

# =====================================================================
elif menu == "Data Analysis":
    st.title("Exploratory Data Analysis")
    df = st.session_state.data
    if df.empty:
        st.error("The active dataset has no rows.")
    elif not labelled(df):
        st.warning("The active dataset has no Academic_Status column, so class-based charts are unavailable. "
                   "Showing descriptive statistics only.")
        st.dataframe(df[FEATURES].describe().T.round(2), use_container_width=True)
    else:
        st.subheader("Descriptive statistics")
        st.dataframe(eda.descriptive_stats(df), use_container_width=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Class distribution")
            st.dataframe(eda.class_distribution(df), hide_index=True, use_container_width=True)
        with c2:
            st.plotly_chart(eda.class_distribution_chart(df), use_container_width=True)
        st.subheader("Interactive charts")
        feat = st.selectbox("Feature for histogram and box plot", FEATURES, index=0)
        a, b = st.columns(2)
        a.plotly_chart(eda.histogram_chart(df, feat), use_container_width=True)
        b.plotly_chart(eda.box_chart(df, feat), use_container_width=True)
        sx, sy = st.columns(2)
        x = sx.selectbox("Scatter X", FEATURES, index=0)
        y = sy.selectbox("Scatter Y", FEATURES, index=1)
        st.plotly_chart(eda.scatter_chart(df, x, y), use_container_width=True)
        st.plotly_chart(eda.correlation_chart(df), use_container_width=True)
        st.plotly_chart(eda.importance_chart(meta["feature_importance"]), use_container_width=True)
        top = meta["feature_importance"][:3]
        st.info("Midterm finding: the three most important predictors are "
                + ", ".join(f"**{t['Feature']}** ({t['Importance']:.3f})" for t in top) + ".")

# =====================================================================
elif menu == "Prediction":
    st.title("Predictive Modeling")
    tab1, tab2, tab3 = st.tabs(["Single student", "Batch (CSV)", "Model metrics"])

    with tab1:
        st.info("Enter values for one student. Previous GPA uses the midterm scale: "
                "1.0 is the best and 5.0 the worst. Engagement Score and Submission Burden are "
                "computed automatically, exactly as in the midterm.")
        c1, c2, c3 = st.columns(3)
        vals = {
            "Attendance_Rate": c1.number_input("Attendance rate (%)", 0.0, 100.0, 80.0, 0.1),
            "Quiz_Average": c1.number_input("Quiz average", 0.0, 100.0, 75.0, 0.1),
            "Assignment_Average": c1.number_input("Assignment average", 0.0, 100.0, 75.0, 0.1),
            "Absences": c2.number_input("Absences", 0, 100, 3, 1),
            "Late_Submissions": c2.number_input("Late submissions", 0, 100, 2, 1),
            "Study_Hours": c2.number_input("Weekly study hours", 0.0, 168.0, 10.0, 0.5),
            "Previous_GPA": c3.number_input("Previous GPA (1.0 best - 5.0 worst)", 1.0, 5.0, 2.5, 0.01),
        }
        student_label = c3.text_input("Student ID (optional)", value="")
        if st.button("Predict", type="primary"):
            errs = prediction.input_errors(vals)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                res = prediction.predict_one(model, vals)
                p = res["probability"]
                st.session_state.prediction_log.append({
                    "Timestamp": res["timestamp"], "Student_ID": student_label or "(none)",
                    **vals, "Predicted_Status": res["label"], "At_Risk_Probability": round(p, 3),
                    "Model_Version": meta["model_version"]})
                (st.error if res["label"] == "At Risk" else st.success)(
                    f"Predicted status: **{res['label']}**")
                r1, r2, r3 = st.columns(3)
                r1.metric("Estimated at-risk probability", f"{p:.1%}")
                r2.metric("Risk band", prediction.risk_band(p))
                r3.metric("Model version", meta["model_version"])
                st.write("**Recommended human follow-up:**", prediction.recommended_followup(p))
                st.caption(f"Prediction timestamp: {res['timestamp']}")
                with st.expander("Features passed to the model"):
                    st.dataframe(res["features"], use_container_width=True)
        if st.session_state.prediction_log:
            st.subheader("Predictions this session")
            log = pd.DataFrame(st.session_state.prediction_log)
            st.dataframe(log, use_container_width=True)
            st.download_button("Download prediction log (CSV)", log.to_csv(index=False),
                               "prediction_log.csv", "text/csv")
        st.subheader("Reproduce the midterm's five sample cases")
        if st.button("Score the five midterm sample students"):
            sample = pd.read_csv(os.path.join(BASE_DIR, "data", "five_sample_predictions.csv"))
            rescored = dashboard.score_students(model, prediction.add_engineered_features(sample))
            out = sample[[ID_COL] + BASE_FEATURES].copy()
            out["Midterm_Predicted_Status"] = sample["Predicted_Status"]
            out["Midterm_Probability"] = sample["Estimated_At_Risk_Probability"]
            out["App_Predicted_Status"] = rescored["Predicted_Status"].values
            out["App_Probability"] = rescored["At_Risk_Probability"].values
            out["Match"] = (out["Midterm_Predicted_Status"] == out["App_Predicted_Status"]) & \
                           ((out["Midterm_Probability"] - out["App_Probability"]).abs() < 0.001)
            st.dataframe(out, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("Upload a CSV of students to score (same required columns as the Dataset page).")
        batch = st.file_uploader("Batch CSV", type="csv", key="batch")
        if batch is not None:
            try:
                raw = pd.read_csv(batch)
            except Exception as exc:
                st.error(f"Could not read the file as CSV: {exc}")
                raw = None
            if raw is not None:
                errors, warnings = validate_schema(raw)
                for w in warnings:
                    st.warning(w)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    scored = dashboard.score_students(model, prepare_uploaded(raw))
                    st.dataframe(scored[[ID_COL] + BASE_FEATURES + ["Predicted_Status",
                                 "At_Risk_Probability", "Risk_Band"]], use_container_width=True)
                    st.download_button("Download scored CSV", scored.to_csv(index=False),
                                       "scored_students.csv", "text/csv")

    with tab3:
        st.subheader("Model metrics (midterm test split)")
        st.caption(f"{meta['model_name']} | version {meta['model_version']} | trained "
                   f"{meta['trained_at_utc']} | train n={meta['n_train']}, test n={meta['n_test']}")
        st.dataframe(pd.DataFrame(meta["metrics"]).T.round(3), use_container_width=True)
        cA, cB = st.columns(2)
        cA.plotly_chart(eda.confusion_chart(meta["confusion_matrices"]["Random Forest"],
                                            meta["confusion_labels"], "Random Forest"),
                        use_container_width=True)
        cB.plotly_chart(eda.confusion_chart(meta["confusion_matrices"]["Decision Tree"],
                                            meta["confusion_labels"], "Decision Tree"),
                        use_container_width=True)
        st.caption("Rows = actual, columns = predicted. The Random Forest was the model chosen in the midterm.")

    st.subheader("Limitations")
    for item in prediction.LIMITATIONS:
        st.markdown(f"- {item}")

# =====================================================================
elif menu == "Dashboard":
    st.title("Decision-Support Dashboard")
    df = st.session_state.data
    if df.empty:
        st.error("The active dataset has no rows.")
        st.stop()
    scored = score_cached(df)
    threshold = st.slider("Review threshold (at-risk probability)", 0.1, 0.9, 0.5, 0.05,
                          help="Students at or above this probability enter the review queue.")
    k = dashboard.kpis(scored, threshold)
    cols = st.columns(6)
    for col, (name, value) in zip(cols, k.items()):
        col.metric(name, value)
    st.caption(f"Model {meta['model_version']}. Scoring the bundled dataset reuses students the model was "
               "trained on, so its probabilities look more confident than they would for new students.")

    a, b = st.columns(2)
    a.plotly_chart(dashboard.risk_band_chart(scored), use_container_width=True)
    b.plotly_chart(dashboard.probability_histogram(scored), use_container_width=True)

    st.subheader("Risk / review queue")
    queue = dashboard.review_queue(scored, threshold, st.session_state.decisions)
    st.dataframe(queue, use_container_width=True, hide_index=True)
    st.download_button("Download review queue (CSV)", queue.to_csv(index=False),
                       "review_queue.csv", "text/csv")

    st.subheader("Interpretation")
    if len(queue):
        top = queue.iloc[0]
        st.markdown(
            f"- **{len(queue)}** of {len(scored)} students ({k['Flagged (%)']}%) are at or above the "
            f"{threshold:.0%} threshold. Their average attendance is {k['Avg attendance (flagged)']}%.\n"
            f"- Highest estimated risk: **{top[ID_COL]}** ({top['At_Risk_Probability']:.0%}). "
            "Probabilities rank who to look at first; they do not prove a student will fail.\n"
            "- Lowering the threshold catches more at-risk students (the model misses about 3 in 10 at 0.5) "
            "but adds more students for reviewers to check."
        )
    else:
        st.info("No students reach the current threshold.")

    st.subheader("Human review: record the final decision")
    st.caption("Only an authorized reviewer (adviser, instructor or program coordinator) records decisions. "
               "Decisions here are held in memory for this session and are lost when it ends; "
               "download the log to keep a copy.")
    if len(queue):
        with st.form("decision_form", clear_on_submit=True):
            sid = st.selectbox("Student", queue[ID_COL].tolist())
            reviewer = st.text_input("Reviewer name and role")
            decision = st.selectbox("Decision", dashboard.DECISIONS[1:])
            notes = st.text_area("Notes / rationale")
            if st.form_submit_button("Save decision"):
                if not reviewer.strip():
                    st.error("Reviewer name is required to record a decision.")
                else:
                    dashboard.record_decision(st.session_state.decisions, sid, reviewer, decision, notes)
                    st.success(f"Decision recorded for {sid}.")
                    st.rerun()
    log = dashboard.decisions_table(st.session_state.decisions)
    st.dataframe(log, use_container_width=True, hide_index=True)
    if len(log):
        st.download_button("Download decision log (CSV)", log.to_csv(index=False),
                           "decision_log.csv", "text/csv")

# =====================================================================
else:
    st.title("System Design (SAD Integration)")
    st.subheader("Actors")
    st.table(pd.DataFrame([
        ("Instructor / Adviser", "Views predictions and the review queue for their students; records the follow-up decision."),
        ("Program Coordinator", "Views all results and KPIs; records or overrides final decisions; exports reports."),
        ("System Administrator", "Uploads and validates datasets; maintains the model and deployment. Does not record academic decisions."),
        ("Student", "No access to predictions in this version; receives support only through a human reviewer."),
    ], columns=["Actor", "Role"]))

    st.subheader("Functional requirements")
    st.markdown(
        "1. **FR1 - Dataset upload and validation:** the system accepts a CSV, checks required columns, and reports missing, duplicate and out-of-range values (Dataset page).\n"
        "2. **FR2 - Exploratory analysis:** the system shows descriptive statistics, class distribution and interactive charts (Data Analysis page).\n"
        "3. **FR3 - Risk prediction:** the system predicts the academic status and at-risk probability for one student or a batch using the saved midterm model (Prediction page).\n"
        "4. **FR4 - Review queue:** the system lists students above a chosen threshold, ranked by probability, with a recommended follow-up (Dashboard).\n"
        "5. **FR5 - Human decision recording:** an authorized reviewer records a final decision with notes; the log can be exported (Dashboard)."
    )
    st.subheader("Non-functional requirements")
    st.markdown(
        "1. **NFR1 - Transparency:** model version, prediction timestamp, limitations and the human-review notice are visible with every prediction.\n"
        "2. **NFR2 - Performance:** the model is loaded once with `st.cache_resource` and never retrained on page refresh; pages respond within a few seconds on the 300-row dataset.\n"
        "3. **NFR3 - Privacy and security:** only synthetic or authorized data is used; no credentials or real student records are stored in the repository; decisions are kept in session memory only."
    )

    st.subheader("Updated use case diagram")
    if os.path.exists(DIAGRAM):
        st.image(DIAGRAM, use_container_width=True)
    else:
        st.error("assets/use_case_diagram.png is missing from the repository.")

    st.subheader("Decision workflow")
    st.markdown(
        "1. **Data in:** administrator uploads or refreshes the dataset; the system validates the schema.\n"
        "2. **Clean and engineer:** invalid values become missing; Engagement Score and Submission Burden are computed.\n"
        "3. **Predict:** the saved pipeline imputes missing values and returns a label and at-risk probability.\n"
        "4. **Prioritize:** students at or above the threshold enter the review queue, highest probability first.\n"
        "5. **Human review:** the adviser or instructor checks the student's record and context.\n"
        "6. **Decide and record:** the reviewer chooses a follow-up (monitor, tutoring referral, counseling referral, or recheck data) and records notes.\n"
        "7. **Follow up and feed back:** outcomes are reviewed by the coordinator and can inform future retraining offline."
    )
    st.warning("Predictions are decision-support outputs. They are not automatic authorization to deny a "
               "reservation, penalize a student, or withhold student services.")
