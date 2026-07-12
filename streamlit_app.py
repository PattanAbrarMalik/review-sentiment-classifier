"""
Streamlit UI for the Review Sentiment Classifier.

Run locally:
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sentiment Reader",
    page_icon="📝",
    layout="centered"
)

# Backend API URL
try:
    API_BASE = st.secrets["API_BASE"]
except Exception:
    API_BASE = "https://review-sentiment-classifier-u9u4.onrender.com"

MODEL_OPTIONS = {
    "N-grams": "n_gram",
    "Bag of Words": "model",
    "Binary Bag of Words": "binary_count_vect",
    "TF-IDF": "tf-idf",
}

st.title("📝 Read the Room")
st.caption("Paste a review below. The model predicts its sentiment.")

with st.form("predict_form"):

    text = st.text_area(
        "Review Text",
        placeholder="It never saves my design and the export keeps failing...",
        height=120,
    )

    model_label = st.selectbox(
        "Model",
        list(MODEL_OPTIONS.keys())
    )

    submitted = st.form_submit_button(
        "Read It",
        use_container_width=True
    )

if submitted:

    if not text.strip():
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Reading..."):

        try:
            # Wake up Render (ignore timeout if service is sleeping)
            try:
                requests.get(
                    f"{API_BASE}/health",
                    timeout=60
                )
            except requests.exceptions.RequestException:
                pass

            response = requests.post(
                f"{API_BASE}/predict",
                json={
                    "text": text,
                    "model_name": MODEL_OPTIONS[model_label]
                },
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            probability = data["probability_positive"]
            label = data["label"]

        except requests.exceptions.RequestException as e:

            st.error(
                "The backend is currently unavailable.\n\n"
                "If you're using the free Render plan, "
                "the first request after inactivity may take up to one minute.\n\n"
                f"Error:\n{e}"
            )

            st.stop()

    color = "#5B7A5E" if label == "positive" else "#A2432E"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={
                "suffix": "%",
                "font": {
                    "size": 36
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1
                },
                "bar": {
                    "color": color
                },
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "#F5E9DD"
                    },
                    {
                        "range": [50, 100],
                        "color": "#E9F0E5"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=10
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    verdict = "Positive" if label == "positive" else "Negative"

    st.markdown(f"## {verdict}")

    st.metric(
        label="Positive Probability",
        value=f"{probability}%"
    )