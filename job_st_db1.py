import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="HR Job Acceptance Dashboard",
    page_icon="📊",
    layout="wide"
)


st.title("📊 HR Job Acceptance Prediction Dashboard")
st.markdown("---")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("job_acceptance.csv") 
    return df

df = load_data()

# -------------------------------
# Create Interview Score
# -------------------------------
df["interview_score"] = (
    df["technical_score"] +
    df["aptitude_score"] +
    df["communication_score"]
) / 3

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔍 Filters")

gender = st.sidebar.multiselect(
    "Gender",
    options=df["gender"].unique(),
    placeholder="Select "
)
intv_cat = st.sidebar.multiselect(
    "Interview score",
    options=df["interview_category"].unique(),
    placeholder="Select "
)
skil_per = st.sidebar.multiselect(
    "Skill Percentage",
    options=df["skills_match_level"].unique(),
    placeholder="Select "
)



status = st.sidebar.multiselect(
    "Placement Status",
    options=df["status"].unique(),
    placeholder="Select "
)

# -------------------------------
# KPI Calculations
# -------------------------------
total_candidates = len(df)



avg_interview = df["interview_score"].mean()

avg_skill = df["skills_match_percentage"].mean()

high_risk = df[
    (df["skills_match_percentage"] < 60) &
    (df["interview_score"] < 60)
]

high_risk_percent = (
    len(high_risk) / len(df)
) * 100

# -------------------------------
# KPI Cards
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    " Total Candidates",
    f"{total_candidates:,}"
)

placement_rate = (df["status"] == "placed").mean() * 100
col2.metric("Placement Rate", f"{placement_rate:.2f}%")

col3.metric(
    " Avg Interview",
    f"{avg_interview:.2f}"
)
col4, col5,col6 = st.columns(3)

col4.metric(
    " Avg Skills Match",
    f"{avg_skill:.2f}%"
)
placement_rate = (df["status"] == "not placed").mean() * 100
col5.metric("Dropout Rate", f"{placement_rate:.2f}%")

col6.metric(
    "High-Risk %",
    f"{high_risk_percent:.2f}%"
)
#------------
#filder
#------------
st.subheader(" Filtered Candidate Data")
filtered_df = df.copy()

if gender:
    filtered_df = filtered_df[
        filtered_df["gender"].isin(gender)
    ]

if intv_cat:
    filtered_df = filtered_df[
        filtered_df["interview_category"].isin(intv_cat)
    ]

if skil_per:
    filtered_df = filtered_df[
        filtered_df["skills_match_level"].isin(skil_per)
    ]

if status:
    filtered_df = filtered_df[
        filtered_df["status"].isin(status)
    ]
st.dataframe(
    filtered_df[
        [
            "gender",
            "status",
            "skills_match_percentage",
            "interview_score"
        ]
    ],
    use_container_width=True
)

st.markdown("---")

#-----------
#bestmodel
#-----------
data = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest",
              "SVM", "KNN", "Naive Bayes", "XGBoost"],
    "Accuracy": [0.82, 0.84, 0.89, 0.84, 0.74, 0.85, 0.90],
    "Precision (0)": [0.85, 0.89, 0.90, 0.85, 0.80, 0.88, 0.93],
    "Precision (1)": [0.73, 0.72, 0.85, 0.78, 0.57, 0.76, 0.84],
    "Recall (0)": [0.90, 0.88, 0.94, 0.92, 0.83, 0.90, 0.94],
    "Recall (1)": [0.62, 0.74, 0.75, 0.62, 0.52, 0.72, 0.82],
    "F1 (0)": [0.87, 0.88, 0.92, 0.89, 0.82, 0.89, 0.93],
    "F1 (1)": [0.67, 0.73, 0.80, 0.69, 0.54, 0.74, 0.83],
}


df = pd.DataFrame(data)


def highlight_xgb(row):
    if row["Model"] == "XGBoost":
        return ["background-color: lightgreen; font-weight: bold"] * len(row)
    else:
        return [""] * len(row)

st.title("Model Comparison Table")

st.dataframe(df.style.apply(highlight_xgb, axis=1))
