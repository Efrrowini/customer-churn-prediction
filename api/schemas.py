from pydantic import BaseModel, Field
from typing import List


class CustomerInput(BaseModel):
    tenure: float = Field(12, ge=0, le=72, description="Months as customer")
    MonthlyCharges: float = Field(65.0, ge=0, description="Monthly bill")
    TotalCharges: float = Field(780.0, ge=0, description="Total spend")
    Contract: str = Field("Month-to-month", description="Month-to-month | One year | Two year")
    InternetService: str = Field("Fiber optic", description="DSL | Fiber optic | No")
    PaymentMethod: str = Field("Electronic check")
    TechSupport: str = Field("No", description="Yes | No | No internet service")
    OnlineSecurity: str = Field("No", description="Yes | No | No internet service")
    gender: str = Field("Male", description="Male | Female")
    SeniorCitizen: int = Field(0, ge=0, le=1, description="0 or 1")
    Partner: str = Field("No", description="Yes | No")
    Dependents: str = Field("No", description="Yes | No")
    PhoneService: str = Field("Yes", description="Yes | No")
    MultipleLines: str = Field("No", description="Yes | No | No phone service")
    OnlineBackup: str = Field("No", description="Yes | No | No internet service")
    DeviceProtection: str = Field("No", description="Yes | No | No internet service")
    StreamingTV: str = Field("No", description="Yes | No | No internet service")
    StreamingMovies: str = Field("No", description="Yes | No | No internet service")
    PaperlessBilling: str = Field("Yes", description="Yes | No")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 2,
                "MonthlyCharges": 95.5,
                "TotalCharges": 191.0,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
                "TechSupport": "No",
                "OnlineSecurity": "No",
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "PaperlessBilling": "Yes"
            }
        }


class PredictionOutput(BaseModel):
    churn_probability: float
    risk_level: str
    top_risk_factors: List[str]
    recommendation: str