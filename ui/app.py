"""Streamlit dashboard: single prediction, data insights, bulk upload +
retrain trigger, and model status/uptime."""

import os
import sys
from glob import glob
from pathlib import Path

# Ensure `src` (a sibling of this file's parent) is importable regardless of
# whether Streamlit is launched from the repo root (local dev) or via
# `streamlit run ui/app.py` from /app (Docker) -- Streamlit only adds this
# script's own directory to sys.path by default, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

from src.preprocessing import (
    plot_color_channel_dominance,
    plot_edge_detection,
    plot_class_distribution_and_intensity,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TRAIN_DIR = "data/train"

st.set_page_config(page_title="Candlestick Trend Classifier", layout="wide")
st.title("Candlestick Trend Classifier")
st.caption(
    "Classifies whether a candlestick chart image visually shows a rising "
    "(\"Up\") or falling (\"Down\") trend across the candles drawn in the "
    "image itself -- a visual pattern-recognition task, not a forecast of "
    "where the price goes next."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Single Prediction", "Data Insights", "Bulk Upload & Retrain", "Model Status & Uptime"]
)

# ---------------------------------------------------------------------------
# Tab 1: Single Prediction
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Predict a single candlestick chart image")
    uploaded = st.file_uploader(
        "Upload a chart image", type=["png", "jpg", "jpeg"], key="predict_uploader"
    )
    if uploaded is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", type="primary"):
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    with col2:
                        st.metric("Prediction", result["prediction"])
                        st.progress(
                            result["confidence"],
                            text=f"Confidence: {result['confidence'] * 100:.1f}%",
                        )
                        st.caption(f"Raw Up-probability: {result['raw_probability']}")
                except requests.RequestException as e:
                    st.error(f"Prediction request failed: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Data Insights
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Dataset Insights")
    try:
        insights = requests.get(f"{API_URL}/insights", timeout=10).json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train Up", insights["train"]["Up"])
        col2.metric("Train Down", insights["train"]["Down"])
        col3.metric("Test images", insights["test"]["Up"] + insights["test"]["Down"])
        col4.metric("Pending Uploads", insights["uploads"]["pending_uploads"])
    except requests.RequestException as e:
        st.error(f"Could not reach API: {e}")

    st.divider()
    st.markdown("### Feature Interpretations")

    @st.cache_data(show_spinner="Reading sample images...")
    def _sample_paths(train_dir, sample_size=150):
        return {
            "Up": sorted(glob(f"{train_dir}/Up/*"))[:sample_size],
            "Down": sorted(glob(f"{train_dir}/Down/*"))[:sample_size],
        }

    paths = _sample_paths(TRAIN_DIR)
    if not paths["Up"] or not paths["Down"]:
        st.warning(
            f"No images found under {TRAIN_DIR}/Up or {TRAIN_DIR}/Down "
            "(check the volume mount if running in Docker)."
        )
    else:
        st.markdown("**1. Color Channel Dominance** — visually rising vs falling images")
        fig1, means = plot_color_channel_dominance(paths)
        st.pyplot(fig1)
        st.caption(
            "Bullish (green) candles push price up, bearish (red) candles push it "
            "down, so an image with an overall rising visual trend should tend to "
            "contain more green candles, and vice versa -- the Up bars should skew "
            "green and Down bars should skew red, a direct visual cue the model can "
            "exploit alongside candle shape."
        )

        st.markdown("**2. Edge Detection** — isolating candle wicks and bodies")
        example_class = st.selectbox("Preview class", ["Up", "Down"], key="edge_class")
        example_path = paths[example_class][0]
        fig2, _ = plot_edge_detection(example_path)
        st.pyplot(fig2)
        st.caption(
            "The Sobel map highlights the thin vertical wicks and rectangular "
            "bodies the CNN uses as structural signal, separate from color."
        )

        st.markdown("**3. Class Distribution & Pixel Intensity**")
        fig3, counts = plot_class_distribution_and_intensity(paths)
        st.pyplot(fig3)
        st.caption(
            "Confirms class balance (no heavy skew needing class-weighting) and "
            "whether Up/Down images differ in overall brightness -- if they "
            "didn't, we'd worry the model is learning a brightness shortcut "
            "instead of real pattern structure."
        )

# ---------------------------------------------------------------------------
# Tab 3: Bulk Upload & Retrain Trigger
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Bulk Upload New Training Data")
    label = st.selectbox("Label for this batch", ["Up", "Down"])
    st.caption(
        "Label by what the chart image itself visually shows: \"Up\" if the "
        "candles trend upward left-to-right, \"Down\" if they trend downward."
    )
    bulk_files = st.file_uploader(
        "Upload multiple chart images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="bulk_uploader",
    )
    if st.button("Upload Batch") and bulk_files:
        with st.spinner(f"Uploading {len(bulk_files)} files..."):
            files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
            try:
                resp = requests.post(
                    f"{API_URL}/upload-data",
                    data={"label": label},
                    files=files_payload,
                    timeout=60,
                )
                resp.raise_for_status()
                result = resp.json()
                st.success(
                    f"Saved {result['saved']} images to '{label}'. "
                    f"Pending for retraining: {result['pending_uploads']}"
                )
                if result.get("auto_retrain_recommended"):
                    st.warning(
                        f"{result['pending_uploads']} samples pending -- "
                        "consider retraining now."
                    )
            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("Trigger Model Retraining")
    st.caption(
        "Fine-tunes the current model on data/train/ (including newly uploaded "
        "samples) with a lowered learning rate and early stopping."
    )
    if st.button("Trigger Model Retraining", type="primary"):
        with st.spinner("Retraining in progress -- this can take a minute..."):
            try:
                resp = requests.post(f"{API_URL}/retrain", timeout=600)
                resp.raise_for_status()
                result = resp.json()
                st.success(
                    f"Retraining complete. Model version {result['model_version']}. "
                    f"Samples used: {result['samples_used']}"
                )
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**Metrics before**")
                    st.json(result["metrics_before"])
                with colB:
                    st.markdown("**Metrics after**")
                    st.json(result["metrics_after"])
            except requests.RequestException as e:
                st.error(f"Retraining failed: {e}")

# ---------------------------------------------------------------------------
# Tab 4: Model Status & Uptime
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Model Status & Uptime")
    try:
        health = requests.get(f"{API_URL}/health", timeout=10).json()
        up = requests.get(f"{API_URL}/uptime", timeout=10).json()

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", health["status"].upper())
        col2.metric("Model Loaded", "Yes" if health["model_loaded"] else "No")
        col3.metric("Currently Retraining", "Yes" if health["is_retraining"] else "No")

        uptime_seconds = up["uptime_seconds"]
        hrs, rem = divmod(uptime_seconds, 3600)
        mins, secs = divmod(rem, 60)
        st.metric("API Uptime", f"{int(hrs)}h {int(mins)}m {int(secs)}s")
        st.write(f"**Started at:** {up['started_at']}")
        st.write(f"**Model version:** {up['model_version']}")
        st.write(f"**Last retrained at:** {up['last_trained_at'] or 'Never (initial notebook-trained model)'}")
    except requests.RequestException as e:
        st.error(f"Could not reach API: {e}")

    if st.button("Refresh"):
        st.rerun()
