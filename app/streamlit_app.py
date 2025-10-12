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
threshold_artifact = float(np.load(THRESHOLD_PATH)[0])
with open(TRAINCOLS_PATH, "r", encoding="utf-8") as f:
    train_cols = json.load(f)

st.title("Telco Customer Churn – XGBoost")
st.caption(f"Final model • AUC≈0.84 • Operating threshold (artifact) = {threshold_artifact}")
st.success("✅ Model and artifacts loaded successfully.")

# ========= UI =========
st.subheader("Customer Profile & Services")

gender_opts   = ["Female", "Male"]  # kept for UX (even if 'gender' not used in X)
yn_opts       = ["No", "Yes"]
internet_opts = ["DSL", "Fiber optic", "No internet service"]
contract_opts = ["Month-to-month", "One year", "Two year"]
payment_opts  = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
]
tri_opts      = ["No", "Yes", "No internet service"]  # for 3-state service fields

c1, c2, c3 = st.columns(3)
with c1:
    gender      = st.selectbox("Gender (not used in model)", gender_opts, index=gender_opts.index("Male"))
    senior      = st.selectbox("Senior Citizen", yn_opts, index=yn_opts.index("No"))
    partner     = st.selectbox("Partner", yn_opts, index=yn_opts.index("No"))
    dependents  = st.selectbox("Dependents", yn_opts, index=yn_opts.index("No"))
with c2:
    tenure      = st.number_input("Tenure (months)", min_value=0, max_value=72, value=2, step=1)
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

st.divider()

# Optional: allow experimenting different thresholds in UI (prediction text uses this)
t_ui = st.slider("🔧 Try a different threshold (for what-if view)", 0.05, 0.95, threshold_artifact, 0.01)

# ========= Feature construction (must mirror training) =========
# We DO NOT feed 'gender' or 'TotalCharges' because training X dropped them.
# But we still compute TotalCharges for ChargeRatio.
total_charges = float(monthly) * max(int(tenure), 1)
charge_ratio  = total_charges / (int(tenure) + 1)  # same as training: /(tenure+1)

# PaymentAutomatic derived from PaymentMethod (case-insensitive)
payment_automatic = 1 if "automatic" in payment.lower() else 0

# Helper: 3-state to binary or one-hots, depending on training columns
def three_state_to_raw(val: str, base: str, raw: dict):
    """
    If training had simple binary (0/1), we keep base as 0/1 with 'Yes'=1 else 0
    (so 'No internet service' -> 0).
    If training had one-hots for the three states, we set them accordingly.
    """
    # default simple binary
    raw[base] = 1 if val == "Yes" else 0

    # override with detailed one-hots if present in training schema
    keys = [f"{base}_Yes", f"{base}_No", f"{base}_No internet service"]
    if any(k in train_cols for k in keys):
        raw.pop(base, None)
        raw[f"{base}_Yes"] = 1 if val == "Yes" else 0
        raw[f"{base}_No"]  = 1 if val == "No" else 0
        if f"{base}_No internet service" in train_cols:
            raw[f"{base}_No internet service"] = 1 if val == "No internet service" else 0

raw = {
    # numeric / binary (used by model)
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": int(tenure),
    "MonthlyCharges": float(monthly),
    "PaperlessBilling": 1 if paperless == "Yes" else 0,
    "PhoneService": 1 if phoneservice == "Yes" else 0,
    "MultipleLines": 1 if multiplelines == "Yes" else 0,

    # engineered features in your latest training
    "ChargeRatio": float(charge_ratio),
    "PaymentAutomatic": int(payment_automatic),
}

# engineered tenure milestones
for m in [12, 24, 36, 48, 60]:
    raw[f"tenure_ge_{m}m"] = 1 if int(tenure) >= m else 0

# tri-state service features
three_state_to_raw(onlinesec,   "OnlineSecurity",  raw)
three_state_to_raw(onlinebackup,"OnlineBackup",    raw)
three_state_to_raw(deviceprot,  "DeviceProtection",raw)
three_state_to_raw(techsupport, "TechSupport",     raw)

# one-hot for InternetService / Contract / PaymentMethod (drop_first happened at training; reindex will handle)
def set_one_hot(prefix: str, value: str, raw: dict):
    # create all possible dummies you expect; reindex will drop the rest
    raw[f"{prefix}_DSL"] = 1 if value == "DSL" else 0
    raw[f"{prefix}_Fiber optic"] = 1 if value == "Fiber optic" else 0
    raw[f"{prefix}_No internet service"] = 1 if value == "No internet service" else 0

set_one_hot("InternetService", internet, raw)

raw["Contract_One year"] = 1 if contract == "One year" else 0
raw["Contract_Two year"] = 1 if contract == "Two year" else 0
# baseline is Month-to-month

raw["PaymentMethod_Electronic check"]      = 1 if payment == "Electronic check" else 0
raw["PaymentMethod_Mailed check"]          = 1 if payment == "Mailed check" else 0
raw["PaymentMethod_Bank transfer (automatic)"] = 1 if payment == "Bank transfer (automatic)" else 0
raw["PaymentMethod_Credit card (automatic)"]   = 1 if payment == "Credit card (automatic)" else 0

# ========= Align to training columns =========
x = pd.DataFrame([raw])
x = x.reindex(columns=train_cols, fill_value=0)

with st.expander("🔎 View model input vector"):
    st.dataframe(x, use_container_width=True)

# ========= Predict =========
proba = float(model.predict_proba(x)[:, 1][0])
pred_artifact_thr  = int(proba >= threshold_artifact)
pred_trial_thr     = int(proba >= t_ui)

st.subheader(f"Churn Probability: {proba:.2%}")
st.write(f"• Prediction @ artifact threshold **{threshold_artifact:.2f}**: "
         f"**{'Churn' if pred_artifact_thr==1 else 'Stay'}**")
if abs(t_ui - threshold_artifact) > 1e-9:
    st.write(f"• What-if @ threshold **{t_ui:.2f}**: "
             f"**{'Churn' if pred_trial_thr==1 else 'Stay'}**")

st.caption("Adjust fields to see probability shifts. Feature set includes ChargeRatio, PaymentAutomatic, and tenure milestone flags. "
           "Input vector is automatically aligned to the training schema.")
