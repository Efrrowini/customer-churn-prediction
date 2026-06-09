import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocess import NUM_COLS, CAT_COLS

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load("models/churn_pipeline.pkl")
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    return pipeline, explainer

pipeline, explainer = load_model()
FEATURE_COLS = NUM_COLS + CAT_COLS

# ── Header ───────────────────────────────────────────────
st.title("📊 Customer Churn Risk Predictor")
st.markdown("_Powered by XGBoost + SHAP — enter customer details to get an instant churn risk score with AI explanations._")
st.divider()

# ── Sidebar inputs ───────────────────────────────────────
st.sidebar.header("Customer Profile")
st.sidebar.markdown("Adjust the sliders and dropdowns to match the customer.")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 2)
monthly = st.sidebar.slider("Monthly charges ($)", 18, 120, 95)
total = tenure * monthly

contract = st.sidebar.selectbox("Contract type",
    ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet service",
    ["Fiber optic", "DSL", "No"])
tech_support = st.sidebar.selectbox("Tech support",
    ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Online security",
    ["No", "Yes", "No internet service"])
payment = st.sidebar.selectbox("Payment method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"])
paperless = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])
senior = st.sidebar.selectbox("Senior citizen", ["No", "Yes"])

# ── Build input dataframe ────────────────────────────────
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': float(total),
    'ChargesPerMonth': total / (tenure + 1),
    'IsNewCustomer': int(tenure <= 3),
    'HighSpender': int(monthly > 70),
    'Contract': contract,
    'InternetService': internet,
    'PaymentMethod': payment,
    'TechSupport': tech_support,
    'OnlineSecurity': online_security,
    'gender': 'Male',
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 'No', 'Dependents': 'No',
    'PhoneService': 'Yes', 'MultipleLines': 'No',
    'OnlineBackup': 'No', 'DeviceProtection': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'PaperlessBilling': paperless
}
df_input = pd.DataFrame([input_data])[FEATURE_COLS]

# ── Predict ──────────────────────────────────────────────
proba = float(pipeline.predict_proba(df_input)[0, 1])

# ── Risk colour ──────────────────────────────────────────
if proba >= 0.7:
    risk_label = "High Risk"
    risk_colour = "#ef4444"
    emoji = "🔴"
elif proba >= 0.4:
    risk_label = "Medium Risk"
    risk_colour = "#f59e0b"
    emoji = "🟡"
else:
    risk_label = "Low Risk"
    risk_colour = "#10b981"
    emoji = "🟢"

# ── Main layout ──────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{proba:.1%}")

with col2:
    st.metric("Risk Level", f"{emoji} {risk_label}")
with col3:
    st.metric("Tenure", f"{tenure} months",
              delta="New customer" if tenure <= 3 else "Established")

# ── Risk bar ─────────────────────────────────────────────
st.markdown(f"""
<div style="margin:16px 0 8px;font-size:13px;color:#6b7280">Churn probability</div>
<div style="background:#e5e7eb;border-radius:8px;height:16px;overflow:hidden">
  <div style="width:{proba*100:.1f}%;height:16px;background:{risk_colour};
              border-radius:8px;transition:width .3s"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:11px;
            color:#9ca3af;margin-top:4px">
  <span>0%</span><span>50%</span><span>100%</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── SHAP explanation ─────────────────────────────────────
col_shap, col_rec = st.columns([3, 2])

with col_shap:
    st.subheader("Why is this customer at risk?")

    X_transformed = pipeline.named_steps["preprocessor"].transform(df_input)
    shap_vals = explainer.shap_values(X_transformed)[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    for i, idx in enumerate(top_idx):
        feat = FEATURE_COLS[idx]
        val = shap_vals[idx]
        bar_color = "#ef4444" if val > 0 else "#10b981"
        direction = "pushes toward churn" if val > 0 else "reduces churn risk"
        bar_w = min(abs(val) * 300, 100)
        st.markdown(f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
    <span style="font-weight:500">{feat}</span>
    <span style="color:#6b7280;font-size:11px">{direction}</span>
  </div>
  <div style="background:#e5e7eb;border-radius:4px;height:8px">
    <div style="width:{bar_w:.1f}%;height:8px;background:{bar_color};border-radius:4px"></div>
  </div>
</div>""", unsafe_allow_html=True)

with col_rec:
    st.subheader("Recommendation")
    if proba >= 0.7:
        st.error("**Immediate action needed**")
        st.markdown("""
- Offer contract upgrade (month-to-month -> 1 year)
- Apply loyalty discount (10-15%)
                    - Assign dedicated support agent
- Follow up within 48 hours
        """)
    elif proba >= 0.4:
        st.warning("**Monitor closely**")
        st.markdown("""
- Schedule proactive check-in call
- Offer tech support bundle
- Send satisfaction survey
        """)
    else:
        st.success("**Low risk — no action needed**")
        st.markdown("""
- Customer is stable
- Consider upsell opportunity
        """)

st.divider()
st.caption("Model: XGBoost | CV AUC: 0.845 | Dataset: IBM Telco (7,032 customers) | Built by Efro")