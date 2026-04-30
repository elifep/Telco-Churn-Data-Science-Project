import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from app.schemas import CustomerData, PredictionResponse


MODEL_PATH = "models/random_forest_churn_pipeline.pkl"

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Random Forest modeli ile müşteri kaybı tahmini yapan FastAPI servisi.",
    version="1.0.0"
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH}. Önce 'python train_model.py' çalıştır."
        )

    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/")
def root():
    return {
        "message": "Telco Customer Churn Prediction API çalışıyor.",
        "model": "Random Forest",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": "Random Forest"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        input_df = pd.DataFrame([customer.model_dump()])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "churn": "Yes" if int(prediction) == 1 else "No",
            "churn_probability": round(float(probability), 4),
            "model": "Random Forest"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))