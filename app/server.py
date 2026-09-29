from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODELS_PATH = Path(__file__).resolve().parent / "models"
FEATURE_COLUMNS = [
    "Gender", "Age", "Height", "Weight", "family_history_with_overweight",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE", "CH2O", "SCC", "FAF",
    "TUE", "CALC", "MTRANS",
]
CATEGORICAL_COLUMNS = [
    "Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE",
    "SCC", "CALC", "MTRANS",
]

svm = joblib.load(MODELS_PATH / "svm_model.pkl")
scaler = joblib.load(MODELS_PATH / "scaler.pkl")
encoders = joblib.load(MODELS_PATH / "label_encoder.pkl")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Gender: Literal["Female", "Male"]
    Age: float = Field(gt=0, le=120)
    Height: float = Field(gt=0, le=2.8)
    Weight: float = Field(gt=0, le=500)
    family_history_with_overweight: Literal["yes", "no"]
    FAVC: Literal["yes", "no"]
    FCVC: float = Field(ge=0, le=5)
    NCP: float = Field(ge=0, le=10)
    CAEC: Literal["no", "Sometimes", "Frequently", "Always"]
    SMOKE: Literal["yes", "no"]
    CH2O: float = Field(ge=0, le=5)
    SCC: Literal["yes", "no"]
    FAF: float = Field(ge=0, le=10)
    TUE: float = Field(ge=0, le=24)
    CALC: Literal["no", "Sometimes", "Frequently", "Always"]
    MTRANS: Literal[
        "Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"
    ]


app = FastAPI(title="Obesity Levels Prediction API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Model API"}


@app.post("/predict")
def predict(data: PredictionRequest) -> dict[str, list[dict[str, object]]]:
    new_data = pd.DataFrame([data.model_dump()])[FEATURE_COLUMNS]
    for column in CATEGORICAL_COLUMNS:
        new_data[column] = encoders[column].transform(new_data[column])

    new_data_scaled = scaler.transform(new_data)
    prediction = svm.predict(new_data_scaled)
    probabilities = svm.predict_proba(new_data_scaled)
    target_encoder = encoders["NObeyesdad"]
    results = []
    for index, predicted_value in enumerate(prediction):
        pred_class = target_encoder.inverse_transform([predicted_value])[0]
        confidence = float(probabilities[index][predicted_value])
        results.append({
            "predicted_class": pred_class,
            "confidence": round(confidence, 2),
        })

    return {"results": results}