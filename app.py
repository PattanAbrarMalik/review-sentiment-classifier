import os
import config
from fastapi import FastAPI
from pydantic import BaseModel
from Source.utils import load_file
from Source.processing import process_text

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


@app.post("/predict")
def predict(req: PredictRequest):

    vectorizer, model = get_model(req.model_name)

    processed_text = process_text(req.text, config.stem)

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


@app.get("/health")
def health():
    return {
        "status": "ok"
    }