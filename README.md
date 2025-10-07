# Telco Customer Churn Prediction (XGBoost)

Interactive churn prediction for a telco dataset with an end-to-end ML pipeline: preprocessing → model selection → hyperparameter tuning → threshold calibration → deployment (Streamlit) + business dashboard (Power BI).

**Final model**: XGBoost  
**Threshold**: 0.4 (business-driven, maximize recall for retention)  
**Key metrics (test)**: Precision ≈ 0.50 • Recall ≈ 0.86 • F1 ≈ 0.63 • ROC-AUC ≈ 0.84

## 🚀 Quickstart (Local)

```bash
# 1) create env (optional)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) run app
streamlit run app/streamlit_app.py

```
Artifacts are already included in artifacts/.
If you retrain, re-export the three files again:

- final_xgb_model.joblib

- final_threshold.npy

- train_columns.json (must match the training feature order)

## 🌐 Deploy
**Option A — Streamlit Community Cloud** 

1. Push this repo to GitHub (public).

2. Go to share.streamlit.io → “New app” → select your repo.

3. Set Main file path: app/streamlit_app.py.

4. Add requirements.txt.

**Option B — Hugging Face Spaces**

1. Create a new Space → Streamlit template.

2. Upload repo (or connect Git).

3. Set app/streamlit_app.py as the entry point.

## 📊 Model & Business Summary

- Goal: Identify customers likely to churn and prioritize retention offers.

- Model: XGBoost tuned via RandomizedSearchCV + Stratified K-Fold CV.

- Threshold 0.4: chosen to maximize recall (catch more true churners), which aligns with retention use-cases.

**Top drivers (typical on Telco dataset):**

- Contract type (Month-to-month vs. 1–2 year)

- Tenure (low tenure → higher risk)

- Monthly charges (higher → higher risk)

- Internet service (Fiber optic group shows higher churn)

- Tech support (No support → higher churn)

Suggested actions:

- Lock-in offers for month-to-month / short-tenure customers

- Improve fiber-optic satisfaction touchpoints

- Incentivize value-add services (e.g., tech support)

## 🧪 Reproducibility

**Open the notebook:**
```
src/TelcoChurnPrediction.ipynb
```
**What it contains:**

- Data cleaning & encoding

- Train/test split (stratified)

- Models compared: Logistic Regression, Random Forest, XGBoost

- Hyperparameter tuning (RandomizedSearchCV)

- Threshold sweep (0.3–0.7) and selection (0.4)

- Export artifacts:
    ```py
    import os, json, joblib, numpy as np
    os.makedirs("../artifacts", exist_ok=True)

    joblib.dump(final_xgb, "../artifacts/final_xgb_model.joblib")
    np.save("../artifacts/final_threshold.npy", np.array([0.4]))
    json.dump(X_train.columns.tolist(), open("../artifacts/train_columns.json","w"))
    ```
## 📊 Power BI Dashboard for Insight

<p align="center">
  <img src="image.png" alt="Power BI Dashboard" width="950">
</p>