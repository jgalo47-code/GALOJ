# Predictive Student Academic Risk Monitoring (Streamlit)

Systems Analysis and Design - Laboratory Exercise: deploying midterm data-mining outputs.

The app deploys the **Random Forest pipeline from the midterm examination** as a decision-support
tool. A prediction never decides anything by itself: an authorized human reviewer records the final decision.

> **Continuity note.** The lab sheet lists "Campus Facility Reservation Monitoring" for Section A. The midterm
> work actually submitted was **Predictive Student Academic Risk Monitoring**, so this app reuses that exact
> target (`Academic_Status`), features, synthetic dataset, pipeline and results unchanged.

## Modules (sidebar)
| Page | What it does |
|---|---|
| Home | Overview, headline metrics, continuity with the midterm |
| Dataset | CSV upload, schema checks, preview, null / duplicate / out-of-range summary |
| Data Analysis | Descriptive statistics, class distribution, interactive Plotly charts |
| Prediction | Single-student and batch prediction, label + probability, model metrics, limitations |
| Dashboard | KPIs, risk/review queue, interpretation, human decision log |
| System Design | Actors, 5 functional + 3 non-functional requirements, use case diagram, workflow |

## Repository layout
```
app.py                          Streamlit entry point
train_model.py                  Midterm pipeline; run once to rebuild the saved model
requirements.txt
modules/data_processing.py      Midterm generation, cleaning, feature engineering, validation
modules/exploratory_analysis.py EDA charts
modules/prediction.py           Model loading and inference
modules/dashboard.py            KPIs, review queue, decision log
models/trained_model.joblib     Saved preprocessing + Random Forest pipeline
models/model_metadata.json      Version, metrics, features, importance
data/                           Synthetic midterm datasets and the five sample predictions
assets/use_case_diagram.png
scripts/make_use_case_diagram.py  (needs matplotlib; development only)
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The saved model was created with scikit-learn 1.9.1, which `requirements.txt` pins. If you change that
version, rerun `python train_model.py` to regenerate `models/trained_model.joblib`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository (see `.gitignore`; never commit passwords or real student records).
2. Open https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, and set the main file to `app.py`.
4. Under **Advanced settings**, choose Python 3.11 or 3.12, then **Deploy**.
5. If the build fails, open **Manage app > Logs** and check `requirements.txt` first.

## Data and ethics
Only synthetic data is bundled. Do not upload real student records to a public deployment.
Predictions are decision-support outputs, not authorization to deny services or penalize a student.
