# app/streamlit_app.py  (REPLACE ALL)
import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn – XGBoost", page_icon="📡")

# ---------- Robust paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH     = os.path.join(ARTIFACTS_DIR, "final_xgb_model.joblib")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "final_threshold.npy")
TRAINCOLS_PATH = os.path.join(ARTIFACTS_DIR, "train_columns.json")

missing = [p for p in [MODEL_PATH, THRESHOLD_PATH, TRAINCOLS_PATH] if not os.path.exists(p)]
if missing:
    st.error("❌ Missing artifacts:\n" + "\n".join(missing))
    st.stop()

model = joblib.load(MODEL_PATH)
threshold = float(np.load(THRESHOLD_PATH)[0])
with open(TRAINCOLS_PATH, "r", encoding="utf-8") as f:
    train_cols = json.load(f)

st.title("(XGBoost)")
st.caption("Final model @ threshold = 0.4 | Precision≈0.50, Recall≈0.86, F1≈0.63, ROC-AUC≈0.84")
st.success("✅ Model and artifacts loaded successfully.")

# ---------- UI: defaults exactly like your screenshot ----------
st.subheader("Customer Profile & Services")

# lists for deterministic default selection
gender_opts   = ["Female", "Male"]
yn_opts       = ["No", "Yes"]
internet_opts = ["DSL", "Fiber optic", "No internet service"]
contract_opts = ["Month-to-month", "One year", "Two year"]
payment_opts  = ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"]
tri_opts      = ["No", "Yes", "No internet service"]  # for 3-state service fields

c1, c2, c3 = st.columns(3)
with c1:
    gender      = st.selectbox("Gender", gender_opts, index=gender_opts.index("Male"))
    senior      = st.selectbox("Senior Citizen", yn_opts, index=yn_opts.index("No"))
    partner     = st.selectbox("Partner", yn_opts, index=yn_opts.index("No"))
    dependents  = st.selectbox("Dependents", yn_opts, index=yn_opts.index("No"))
with c2:
    tenure      = st.number_input("Tenure (months)", min_value=0, max_value=72, value=2)        # default 2
    monthly     = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=100.0, step=1.0)
    paperless   = st.selectbox("Paperless Billing", yn_opts, index=yn_opts.index("Yes"))
    contract    = st.selectbox("Contract", contract_opts, index=contract_opts.index("Month-to-month"))
with c3:
    internet     = st.selectbox("Internet Service", internet_opts, index=internet_opts.index("Fiber optic"))
    phoneservice = st.selectbox("Phone Service", yn_opts, index=yn_opts.index("No"))
    multiplelines= st.selectbox("Multiple Lines", yn_opts, index=yn_opts.index("Yes"))
    payment      = st.selectbox("Payment Method", payment_opts, index=payment_opts.index("Electronic check"))

c4, c5, c6, c7 = st.columns(4)
with c4:
    onlinesec    = st.selectbox("Online Security", tri_opts, index=tri_opts.index("No"))
with c5:
    onlinebackup = st.selectbox("Online Backup", tri_opts, index=tri_opts.index("No internet service"))
with c6:
    deviceprot   = st.selectbox("Device Protection", tri_opts, index=tri_opts.index("No internet service"))
with c7:
    techsupport  = st.selectbox("Tech Support", tri_opts, index=tri_opts.index("Yes"))

# ---------- build raw dict (numeric + one-hot flags like training) ----------
total_charges = float(monthly) * max(int(tenure), 1)

raw = {
    # if your training used one-hot like gender_Male instead of a single "gender" flag,
    # this will still be safe because reindex will drop/zero missing columns later.
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,

    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total_charges,

    "PhoneService": 1 if phoneservice == "Yes" else 0,
    "MultipleLines": 1 if multiplelines == "Yes" else 0,

    "PaperlessBilling": 1 if paperless == "Yes" else 0,

    # One-hot InternetService
    "InternetService_DSL": 1 if internet == "DSL" else 0,
    "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
    # "No internet service" -> both zero

    # One-hot Contract
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    # One-hot PaymentMethod
    "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
    "PaymentMethod_Bank transfer (automatic)": 1 if payment == "Bank transfer (automatic)" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
}

# handle 3-state service fields to match training encoding
def three_state_to_flags(val: str, base: str):
    """Map 'Yes'/'No'/'No internet service' to detailed one-hots if they exist in training columns."""
    # simple flag (in case training used just 0/1)
    raw[base] = 1 if val == "Yes" else 0

    # if training had detailed one-hots, switch to them
    if (f"{base}_Yes" in train_cols) or (f"{base}_No" in train_cols) or (f"{base}_No internet service" in train_cols):
        raw.pop(base, None)
        raw[f"{base}_Yes"] = 1 if val == "Yes" else 0
        raw[f"{base}_No"] = 1 if val == "No" else 0
        if f"{base}_No internet service" in train_cols:
            raw[f"{base}_No internet service"] = 1 if val == "No internet service" else 0

three_state_to_flags(onlinesec,   "OnlineSecurity")
three_state_to_flags(onlinebackup,"OnlineBackup")
three_state_to_flags(deviceprot,  "DeviceProtection")
three_state_to_flags(techsupport, "TechSupport")

# ---------- align to training columns ----------
x = pd.DataFrame([raw])
x = x.reindex(columns=train_cols, fill_value=0)

with st.expander("🔎 View model input vector"):
    st.dataframe(x, use_container_width=True)

# ---------- predict ----------
proba = float(model.predict_proba(x)[:, 1][0])
pred  = int(proba >= threshold)

st.subheader(f"Churn Probability: {proba:.2%}")
st.write(f"Prediction @ threshold 0.4: **{'Churn' if pred==1 else 'Stay'}**")

st.caption("Adjust fields and observe probability changes. Defaults are set to your requested scenario.")
