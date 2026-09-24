# 1 is good(lower risk) 0 is bad(higher risk)

import streamlit as st
import pandas as pd
import joblib

from db import init_db, log_prediction, SessionLocal, PredictionLog

MODEL_VERSION = "extra_trees_credit_model.pkl" 

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="\U0001F4B3",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    section[data-testid="stSidebar"] { background-color: #111827; }

    h1, h2, h3, p, label, span, div { color: #e5e7eb; }

    div[data-testid="stMetric"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.1s ease-in-out;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        color: white;
        border: none;
    }

    .result-card {
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
        border: 1px solid;
    }
    .result-good {
        background-color: rgba(16, 185, 129, 0.12);
        border-color: #10b981;
    }
    .result-bad {
        background-color: rgba(239, 68, 68, 0.12);
        border-color: #ef4444;
    }
    .result-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

model = joblib.load(MODEL_VERSION)
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account']}

# Create the prediction_log table on first run if it doesn't exist yet.
# If Postgres isn't reachable, keep the app usable but disable logging/history
# rather than crashing the whole page for the applicant.
db_available = True
try:
    init_db()
except Exception as exc:
    db_available = False
    st.warning(f"Database unavailable \u2014 predictions won't be logged this session. ({exc})")

# --- Header ------------------------------------------------------------
st.title("\U0001F4B3 Credit Risk Predictor")
st.caption("Enter the applicant's details to estimate whether their credit risk is GOOD or BAD.")
st.divider()

# --- Inputs (sidebar) ----------------------------------------------------
with st.sidebar:
    st.header("Applicant Details")

    st.subheader("\U0001F464 Personal")
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    job = st.number_input("Job skill level (0-3)", min_value=0, max_value=3, value=1)

    st.subheader("\U0001F3E0 Housing & Savings")
    housing = st.selectbox("Housing", ['own', 'rent', 'free'])
    saving_accounts = st.selectbox("Savings account", ['little', 'moderate', 'rich', 'quite rich'])
    cheking_account = st.selectbox("Checking account", ['little', 'moderate', 'rich'])

    st.subheader("\U0001F4B0 Loan")
    credit_amount = st.number_input("Credit amount", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=1, value=12)

    predict_clicked = st.button("\U0001F50D Predict Risk")

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex.lower()])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([cheking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# --- Summary metrics (always visible) ------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Age", age)
m2.metric("Credit Amount", f"${credit_amount:,.0f}")
m3.metric("Duration", f"{duration} mo")
m4.metric("Job Level", job)

# --- Prediction ------------------------------------------------------------
if predict_clicked:
    pred = model.predict(input_df)[0]
    prob_good = model.predict_proba(input_df)[0][1]
    label = "GOOD" if pred == 1 else "BAD"
    confidence = prob_good if pred == 1 else 1 - prob_good

    st.subheader("Result")
    css_class = "result-good" if pred == 1 else "result-bad"
    icon = "\u2705" if pred == 1 else "\u26A0\uFE0F"
    st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-title">{icon} Predicted Credit Risk: {label}</div>
            <div>Model confidence: {confidence:.0%}</div>
        </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.progress(float(prob_good), text=f"P(good credit) = {prob_good:.0%}")

    if db_available:
        logged = log_prediction(
            age=age,
            sex=sex,
            job=job,
            housing=housing,
            saving_accounts=saving_accounts,
            checking_account=cheking_account,
            credit_amount=credit_amount,
            duration=duration,
            predicted_label=label,
            probability_good=float(prob_good),
            model_version=MODEL_VERSION,
        )
        if not logged:
            st.caption("\u26A0\uFE0F This prediction could not be saved to the audit log.")
else:
    st.info("\U0001F448 Fill in the applicant's details in the sidebar, then click **Predict Risk**.")

# --- Recent predictions (audit trail) ---
if db_available:
    st.divider()
    with st.expander("\U0001F4DC Recent predictions"):
        session = SessionLocal()
        try:
            rows = (
                session.query(PredictionLog)
                .order_by(PredictionLog.created_at.desc())
                .limit(10)
                .all()
            )
            if rows:
                history_df = pd.DataFrame([{
                    "Time (UTC)": r.created_at,
                    "Age": r.age,
                    "Sex": r.sex,
                    "Housing": r.housing,
                    "Credit amount": r.credit_amount,
                    "Duration": r.duration,
                    "Prediction": r.predicted_label,
                    "P(good)": round(r.probability_good, 2),
                } for r in rows])
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No predictions logged yet.")
        finally:
            session.close()