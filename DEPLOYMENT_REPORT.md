# Deployment Report

**Course:** Systems Analysis and Design | **Project:** Predictive Student Academic Risk Monitoring
**GitHub repository:** _<paste URL>_ | **Streamlit app:** _<paste URL>_

## 1. Reused midterm code
- Synthetic dataset generation (seed 42, 300 students) and the rule-based `Academic_Status` label.
- Cleaning: title-casing the target, de-duplication, numeric coercion, range validation to missing values.
- Feature engineering: `Engagement_Score` (0.40/0.30/0.30 weights) and `Submission_Burden`.
- Split (80/20, stratified, seed 42), median imputation, Decision Tree vs Random Forest, class weights.
- Random Forest chosen; metrics, confusion matrices, feature importance and the five sample students.

## 2. Changes made for deployment
- Code split into `modules/` and `train_model.py`; the pipeline is saved once with joblib and only loaded by the app (`st.cache_resource`).
- Added schema validation, error messages, session-only decision log, risk bands and thresholded review queue.
- Midterm project (student academic risk) differs from the lab sheet's Section A topic (facility reservations); the midterm target, features and results were kept.
- `Student_ID` and `Academic_Status` are never passed to the model (no leakage); inference uses exactly the 9 midterm features.

## 3. Tests performed
| Test | Result |
|---|---|
| Re-ran the original midterm script and compared outputs | Cleaned data, metrics and five sample predictions identical |
| Saved model vs midterm on the five sample students | Labels and probabilities match |
| Single prediction input frame has exact training feature order | Pass |
| Out-of-range and missing single-student inputs | Rejected with messages |
| Empty file, missing columns, no-ID / unlabelled file | Error or warning shown as expected |
| Dirty upload (text in numeric column, value 250, blank, duplicate row) | Cleaned, de-duplicated and scored |
| Review queue ordering, KPIs and decision recording | Pass |
| Deployed URL, all six pages, charts, screenshots | _<complete after deployment>_ |

## 4. Limitations
- Synthetic data; the label comes from a formula, so accuracy does not reflect real-world performance.
- At-risk recall about 71%: some at-risk students are missed. Probabilities are uncalibrated.
- The bundled dashboard scores students the model was trained on.
- Decisions are stored only in session memory (lost on restart); download the log to keep it.

## 5. Reflection
_<Write 3-5 sentences: what was hardest about turning a notebook into a system, what the human-review step
changes about how the model should be used, and what you would improve with real, authorized data.>_
