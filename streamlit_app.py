"""
Streamlit UI for the review sentiment model.

Run locally:  streamlit run streamlit_app.py
Deploy free:  push to GitHub, then deploy at https://share.streamlit.io
"""
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Sentiment Reader", page_icon="📝", layout="centered")

# Point this at your deployed FastAPI backend (Cloud Run / Render / Railway URL)
# Locally, falls back to localhost since no secrets.toml exists yet.
try:
    API_BASE = st.secrets["API_BASE"]
except (FileNotFoundError, KeyError):
    API_BASE = "http://localhost:8000"

MODEL_OPTIONS = {
    "N-grams": "n_gram",
    "Bag of words": "model",
    "Binary bag of words": "binary_count_vect",
    "TF-IDF": "tf-idf",
}

st.title("Read the room")
st.caption("Paste a review below. The model scores how positive it reads.")

with st.form("predict_form"):
    text = st.text_area(
        "Review text",
        placeholder="It never saves my design and the export keeps failing...",
        height=120,
    )
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    submitted = st.form_submit_button("Read it", use_container_width=True)

if submitted:
    if not text.strip():
        st.warning("Enter some review text first.")
    else:
        with st.spinner("Reading..."):
            try:
                res = requests.post(
                    f"{API_BASE}/predict",
                    json={"text": text, "model_name": MODEL_OPTIONS[model_label]},
                    timeout=15,
                )
                res.raise_for_status()
                data = res.json()
                prob = data["probability_positive"]
                label = data["label"]
            except requests.exceptions.RequestException as e:
                st.error(f"Couldn't reach the model backend at {API_BASE}. ({e})")
                st.stop()

        color = "#5B7A5E" if label == "positive" else "#A2432E"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#F5E9DD"},
                    {"range": [50, 100], "color": "#E9F0E5"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        verdict = "Positive" if label == "positive" else "Negative"
        st.markdown(f"### {verdict} — {prob}% positive")