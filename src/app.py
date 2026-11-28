from pathlib import Path
from typing import Literal
import joblib
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.joblib")

app = FastAPI(title="Guwahati Speed Parcel Hub API", version="1.0.0")
_model = None


class PredictRequest(BaseModel):
    distance_km: float = Field(..., le=4500)
    weight_kg: float = Field(..., ge=0)
    hub_backlog: int
    pin_code_tier: int
    declared_value_rs: float
    transfers: int
    service: Literal["standard", "express", "economy"]
    destination: Literal["Metro", "Tier-2", "Rural"]


class PredictResponse(BaseModel):
    delivery_days: float


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Champion model not found")
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model = get_model()
    row = pd.DataFrame([payload.model_dump()])
    pred = float(model.predict(row)[0])
    return PredictResponse(delivery_days=round(pred, 4))
