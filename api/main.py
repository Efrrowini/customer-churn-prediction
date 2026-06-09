import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import CustomerInput, PredictionOutput
from src.preprocess import NUM_COLS, CAT_COLS

# ── App setup ────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts telecom customer churn with SHAP explanations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Load model once at startup ───────────────────────────
FEATURE_COLS = NUM_COLS + CAT_COLS
pipeline = joblib.load("models/churn_pipeline.pkl")
explainer = shap.TreeExplainer(pipeline.named_steps["model"])
print("Model and explainer loaded successfully")


def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "High risk"
    elif prob >= 0.4:
        return "Medium risk"
    else:
        return "Low risk"


def get_recommendation(prob: float, top_factors: list) -> str:
    if prob >= 0.7:
        return f"Immediate action needed. Contact customer and offer contract upgrade or discount. Key driver: {top_factors[0]}"
    elif prob >= 0.4:
        return f"Monitor closely. Consider proactive outreach. Key driver: {top_factors[0]}"
    else:
        return "Low churn risk. No immediate action required."
    # ── Endpoints ────────────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": "XGBoost churn predictor",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    # Build input dataframe
    data = customer.model_dump()

    # Add engineered features
    tenure = data["tenure"]
    monthly = data["MonthlyCharges"]
    total = data["TotalCharges"]
    data["ChargesPerMonth"] = total / (tenure + 1)
    data["IsNewCustomer"] = int(tenure <= 3)
    data["HighSpender"] = int(monthly > 70)

    df = pd.DataFrame([data])[FEATURE_COLS]
    # Predict
    proba = float(pipeline.predict_proba(df)[0, 1])

    # SHAP top factors
    X_transformed = pipeline.named_steps["preprocessor"].transform(df)
    shap_vals = explainer.shap_values(X_transformed)[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_factors = [FEATURE_COLS[i] for i in top_idx]

    return PredictionOutput(
        churn_probability=round(proba, 3),
        risk_level=get_risk_level(proba),
        top_risk_factors=top_factors,
        recommendation=get_recommendation(proba, top_factors)
    )