import os
import traceback

import config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Source.processing import process_text
from Source.utils import load_file

app = FastAPI(
    title="Review Sentiment API",
    version="1.0.0",
    description="FastAPI service for Review Sentiment Classification"
)

# Cache loaded models
_cache = {}


def get_model(model_name: str):
    if model_name not in _cache:
        vect_file = os.path.join(config.output_path, f"{model_name}.pkl")
        model_file = os.path.join(config.output_path, f"{model_name}_lr.pkl")

        print(f"Vectorizer path: {os.path.abspath(vect_file)}")
        print(f"Model path: {os.path.abspath(model_file)}")
        print(f"Vectorizer exists: {os.path.exists(vect_file)}")
        print(f"Model exists: {os.path.exists(model_file)}")

        vectorizer = load_file(vect_file)
        model = load_file(model_file)

        _cache[model_name] = (vectorizer, model)

    return _cache[model_name]


class PredictRequest(BaseModel):
    text: str
    model_name: str = "n_gram"


@app.get("/")
def root():
    return {
        "message": "Review Sentiment API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        print("=" * 60)
        print("Prediction request received")
        print(f"Model: {req.model_name}")
        print(f"Text: {req.text}")

        vectorizer, model = get_model(req.model_name)

        processed_text = process_text(req.text, config.stem)
        print(f"Processed text: {processed_text}")

        X = vectorizer.transform([processed_text])

        probability = round(
            float(model.predict_proba(X)[0][1]) * 100,
            2
        )

        return {
            "text": req.text,
            "model_name": req.model_name,
            "probability_positive": probability,
            "label": "positive" if probability >= 50 else "negative"
        }

    except Exception as e:
        print("=" * 60)
        print("ERROR DURING PREDICTION")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail={
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )