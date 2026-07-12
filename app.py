""" 

Minimal FastAPI service around the trained model.
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Then: POST http://localhost:8000/predict  {"text": "...", "model_name": "n_gram"}

import os
import config
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from Source.utils import load_file
from Source.processing import process_text

app = FastAPI(title="Review Sentiment API")

# Serve the frontend at "/" so the API and UI share one deployed URL
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

# Cache loaded models in memory so we don't hit disk on every request
_cache = {}


def get_model(model_name: str):
    if model_name not in _cache:
        vect_file = os.path.join(config.output_path, f"{model_name}.pkl")
        model_file = os.path.join(config.output_path, f"{model_name}_lr.pkl")
        _cache[model_name] = (load_file(vect_file), load_file(model_file))
    return _cache[model_name]


class PredictRequest(BaseModel):
    text: str
    model_name: str = "n_gram"


@app.post("/predict")
def predict(req: PredictRequest):
    vect, model = get_model(req.model_name)
    tokens = [process_text(req.text, config.stem)]
    X = vect.transform(tokens)
    prob = round(float(model.predict_proba(X)[0, 1]) * 100, 2)
    return {
        "text": req.text,
        "model_name": req.model_name,
        "probability_positive": prob,
        "label": "positive" if prob >= 50 else "negative",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
"""

"""
Minimal FastAPI service around the trained model.
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Then: POST http://localhost:8000/predict  {"text": "...", "model_name": "n_gram"}
"""
import os
import config
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from Source.utils import load_file
from Source.processing import process_text

app = FastAPI(title="Review Sentiment API")

# Serve the frontend at "/" so the API and UI share one deployed URL
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

# Cache loaded models in memory so we don't hit disk on every request
_cache = {}


def get_model(model_name: str):
    if model_name not in _cache:
        vect_file = os.path.join(config.output_path, f"{model_name}.pkl")
        model_file = os.path.join(config.output_path, f"{model_name}_lr.pkl")
        _cache[model_name] = (load_file(vect_file), load_file(model_file))
    return _cache[model_name]


class PredictRequest(BaseModel):
    text: str
    model_name: str = "n_gram"


@app.post("/predict")
def predict(req: PredictRequest):
    vect, model = get_model(req.model_name)
    tokens = [process_text(req.text, config.stem)]
    X = vect.transform(tokens)
    prob = round(float(model.predict_proba(X)[0, 1]) * 100, 2)
    return {
        "text": req.text,
        "model_name": req.model_name,
        "probability_positive": prob,
        "label": "positive" if prob >= 50 else "negative",
    }


@app.get("/health")
def health():
    return {"status": "ok"}