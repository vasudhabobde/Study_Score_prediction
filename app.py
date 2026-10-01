import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("best_model.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StudyScore AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------- Main application ---------- */

.stApp {
    background-color: #0f172a;
    color: #f8fafc;
}


/* ---------- Main container ---------- */

.block-container {
    max-width: 1200px;
    padding-top: 35px;
    padding-bottom: 50px;
}


/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #e5e7eb !important;
}


/* ---------- Headings ---------- */

h1 {
    color: #f8fafc !important;
}

h2 {
    color: #f8fafc !important;
}

h3 {
    color: #f8fafc !important;
}


/* ---------- Normal text ---------- */

p {
    color: #cbd5e1;
}


/* ---------- Input labels ---------- */

label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}


/* ---------- Cards using Streamlit containers ---------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #111827;
    border: 1px solid #1e293b;
    border-radius: 18px;
}


/* ---------- Buttons ---------- */

.stButton > button {
    width: 100%;
    min-height: 50px;

    background-color: #2563eb;
    color: white;

    border: none;
    border-radius: 12px;

    font-size: 16px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}


/* ---------- Slider ---------- */

div[data-testid="stSlider"] {
    padding-top: 10px;
}


/* ---------- Number input ---------- */

div[data-baseweb="input"] {
    background-color: #111827;
    border-radius: 10px;
}


/* ---------- Number input text ---------- */

div[data-baseweb="input"] input {
    color: white !important;
}


/* ---------- Metric styling ---------- */

div[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1e293b;
    border-radius: 15px;
    padding: 18px;
}


/* ---------- Progress bar ---------- */

div[data-testid="stProgress"] > div {
    background-color: #1e293b;
}


/* ---------- Alert boxes ---------- */

div[data-testid="stAlert"] {
    border-radius: 12px;
}


/* ---------- Dataframe ---------- */

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}


/* ---------- Horizontal line ---------- */

hr {
    border-color: #1e293b;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 StudyScore AI")

    st.caption("Machine Learning Score Prediction")

    st.divider()

    st.subheader("📌 About")

    st.write(
        "This application predicts an expected exam "
        "score based on the number of hours studied."
    )

    st.divider()

    st.subheader("🤖 Model")

    st.success("Linear Regression")

    st.divider()

    st.subheader("📊 Input")

    st.write("Hours Studied")

    st.divider()

    st.caption("Interactive ML Project")


# ============================================================
# HEADER
# ============================================================

st.title("🎓 StudyScore AI")

st.subheader(
    "Predict your expected exam score using Machine Learning"
)

st.write(
    "Enter the amount of time you studied and let the trained "
    "regression model estimate your exam score."
)


st.divider()


# ============================================================
# MAIN INPUT AREA
# ============================================================

st.header("📚 Study Information")

st.write(
    "Use the slider to select your study time."
)


# ============================================================
# STUDY HOURS SLIDER
# ============================================================

hours = st.slider(
    "📖 Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)


# ============================================================
# MANUAL INPUT
# ============================================================

manual_hours = st.number_input(
    "⌨️ Or enter hours manually",
    min_value=0.0,
    max_value=24.0,
    value=float(hours),
    step=0.5
)


# Use manual input for prediction
hours = manual_hours


# ============================================================
# CURRENT INPUT DISPLAY
# ============================================================

st.info(
    f"📚 Current study time: **{hours:.1f} hours**"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "🚀 Predict My Score",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # Create DataFrame with the same feature name
    # used during model training
    input_data = pd.DataFrame({
        "Hours": [hours]
    })


    # Generate prediction
    prediction = model.predict(input_data)


    # Convert prediction safely to one number
    predicted_score = float(
        np.asarray(prediction).ravel()[0]
    )


    # Keep score within 0-100 for display
    display_score = max(
        0.0,
        min(100.0, predicted_score)
    )


    # ========================================================
    # PERFORMANCE CATEGORY
    # ========================================================

    if display_score >= 90:

        category = "Excellent 🌟"

        recommendation = (
            "Excellent preparation! Continue your current "
            "study routine and maintain regular revision."
        )

    elif display_score >= 75:

        category = "Very Good 🎯"

        recommendation = (
            "Great progress! Continue practicing and use "
            "focused revision to improve further."
        )

    elif display_score >= 60:

        category = "Good 👍"

        recommendation = (
            "You are on the right track. Increasing focused "
            "study time may help improve your expected score."
        )

    elif display_score >= 40:

        category = "Needs Improvement 📚"

        recommendation = (
            "Consider following a structured study schedule "
            "and gradually increasing your study time."
        )

    else:

        category = "More Preparation Needed ⚠️"

        recommendation = (
            "Try building a consistent study routine and "
            "spend more time reviewing important concepts."
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.header("📊 Prediction Result")


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            label="📚 Study Hours",
            value=f"{hours:.1f}"
        )


    with col2:

        st.metric(
            label="📈 Predicted Score",
            value=f"{display_score:.2f}/100"
        )


    with col3:

        st.metric(
            label="🎯 Performance",
            value=category
        )


    # ========================================================
    # LARGE SCORE DISPLAY
    # ========================================================

    st.subheader("🎯 Your Predicted Score")


    score_col1, score_col2, score_col3 = st.columns(
        [1, 2, 1]
    )


    with score_col2:

        st.metric(
            label="Exam Score",
            value=f"{display_score:.2f}",
            delta=f"{display_score:.2f}% estimated"
        )


    # ========================================================
    # PROGRESS BAR
    # ========================================================

    st.subheader("📊 Score Visualization")

    st.progress(
        int(display_score)
    )

    st.caption(
        f"Estimated performance: {display_score:.2f}%"
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.subheader("💡 Study Recommendation")

    st.info(
        recommendation
    )


    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.subheader("📋 Prediction Summary")


    result_df = pd.DataFrame({
        "Parameter": [
            "Hours Studied",
            "Predicted Score",
            "Performance",
            "Model Used"
        ],

        "Value": [
            f"{hours:.1f} hours",
            f"{display_score:.2f} / 100",
            category,
            "Linear Regression"
        ]
    })


    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.header("🧠 How Does It Work?")


col1, col2, col3 = st.columns(3)


with col1:

    st.subheader("1️⃣ Input")

    st.write(
        "Enter the number of hours you studied using "
        "the slider or manual input."
    )


with col2:

    st.subheader("2️⃣ Machine Learning")

    st.write(
        "The trained Linear Regression model analyzes "
        "the study-hours input."
    )


with col3:

    st.subheader("3️⃣ Prediction")

    st.write(
        "The model produces an estimated exam score "
        "for the entered study time."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("🤖 About the Machine Learning Model")

with st.expander("View Model Information"):

    st.write(
        """
        **Algorithm:** Linear Regression

        **Input Feature:** Hours Studied

        **Output:** Predicted Exam Score

        The model was trained using the study-hours and
        exam-score dataset from the machine learning notebook.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 StudyScore AI • Machine Learning Project • Streamlit"
)