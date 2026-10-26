"""Exploratory data analysis helpers: descriptive stats and interactive Plotly charts."""
import pandas as pd
import plotly.express as px

from modules.data_processing import BASE_FEATURES, FEATURES, TARGET

COLORS = {"At Risk": "#d1495b", "Satisfactory": "#2e86ab"}


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].describe().T.round(2)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().rename_axis(TARGET).reset_index(name="Students")
    counts["Share (%)"] = (counts["Students"] / counts["Students"].sum() * 100).round(1)
    return counts


def class_distribution_chart(df: pd.DataFrame):
    data = class_distribution(df)
    return px.bar(data, x=TARGET, y="Students", color=TARGET, color_discrete_map=COLORS,
                  text="Students", title="Class distribution")


def histogram_chart(df: pd.DataFrame, feature: str):
    return px.histogram(df, x=feature, color=TARGET, color_discrete_map=COLORS,
                        barmode="overlay", opacity=0.7, nbins=25,
                        title=f"Distribution of {feature} by status")


def box_chart(df: pd.DataFrame, feature: str):
    return px.box(df, x=TARGET, y=feature, color=TARGET, color_discrete_map=COLORS,
                  title=f"{feature} by status")


def scatter_chart(df: pd.DataFrame, x: str, y: str):
    hover = [c for c in ["Student_ID", "Previous_GPA"] if c in df.columns]
    return px.scatter(df, x=x, y=y, color=TARGET, color_discrete_map=COLORS,
                      hover_data=hover, title=f"{y} vs {x}")


def correlation_chart(df: pd.DataFrame):
    corr = df[FEATURES].corr().round(2)
    return px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, title="Correlation of predictors")


def importance_chart(importance: list):
    imp = pd.DataFrame(importance).sort_values("Importance")
    return px.bar(imp, x="Importance", y="Feature", orientation="h",
                  title="Feature importance (Random Forest, midterm)")


def confusion_chart(matrix: list, labels: list, title: str = "Confusion matrix (midterm test split)"):
    cm = pd.DataFrame(matrix, index=[f"Actual {l}" for l in labels],
                      columns=[f"Predicted {l}" for l in labels])
    return px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title=title)
