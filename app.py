"""
app.py
------
Optional Streamlit web interface for the Network Intrusion Detection model.

Features
--------
- Upload a CSV file of network traffic records.
- Load the trained pipeline from outputs/intrusion_detection_model.joblib.
- Validate the uploaded file schema.
- Predict labels and attack probabilities using the trained model.
- Display a summary of Normal vs Attack predictions.
- Show a styled preview table with colour-coded results.
- Allow the user to download the full prediction results as CSV.

Important notes
---------------
- This interface makes OFFLINE predictions using the trained Random Forest
  model. It does NOT capture live packets or scan real networks.
- Attack probability scores are model estimates and are NOT proof of malicious
  intent. They should be treated as a starting point for further analysis.
- Upload the UNSW-NB15 testing CSV or any CSV with compatible columns.
  The interface will automatically strip 'label', 'id', and 'attack_cat'
  from the input before prediction.

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

MODEL_PATH = Path("outputs") / "intrusion_detection_model.joblib"

# ── Page configuration ────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────── #
st.markdown(
    """
    <style>
    .metric-card { background:#1e1e2e; border-radius:12px; padding:1rem;
                   text-align:center; border:1px solid #313244; }
    .attack-badge { color:#F44336; font-weight:700; }
    .normal-badge { color:#4CAF50; font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────── #


@st.cache_resource(show_spinner="Loading model from disk …")
def _load_pipeline():
    """Load and cache the trained pipeline. Returns None if not found."""
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and lowercase column names — mirrors data_loader.py."""
    df.columns = [c.strip().lower() for c in df.columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _run_prediction(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Predict on a normalized DataFrame and return an annotated output DF."""
    from src.predict import predict_from_df

    results = predict_from_df(pipeline, df)
    out = df.copy()
    out["predicted_label"] = results["y_pred"]
    out["predicted_label_name"] = results["label_names"]
    out["attack_probability"] = results["attack_scores"].round(4)
    return out


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ── UI ────────────────────────────────────────────────────────────────────── #


def main() -> None:
    # ── Sidebar ────────────────────────────────────────────────────────── #
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/"
            "Network-switch.jpg/320px-Network-switch.jpg",
            use_column_width=True,
        )
        st.title("🛡️ Network IDS")
        st.caption("Powered by Random Forest · UNSW-NB15")
        st.divider()
        st.markdown(
            """
            **How to use:**
            1. Ensure the model is trained (`python train_model.py`).
            2. Upload a CSV of network traffic records.
            3. Review the predictions below.

            **Disclaimer:** Model scores are statistical estimates.
            They are not proof of malicious activity and should not be
            used as the sole basis for security decisions.
            """
        )
        st.divider()
        model_status = "✅ Model loaded" if MODEL_PATH.exists() else "❌ No model found"
        st.caption(model_status)

    # ── Main area ──────────────────────────────────────────────────────── #
    st.title("🛡️ Network Intrusion Detection")
    st.subheader("Upload network traffic records for classification")

    # ── Model check ────────────────────────────────────────────────────── #
    pipeline = _load_pipeline()
    if pipeline is None:
        st.error(
            "**Model not found.** Please run the training pipeline first:\n\n"
            "```powershell\npython train_model.py\n```\n\n"
            f"Expected model path: `{MODEL_PATH.resolve()}`"
        )
        st.stop()

    # ── File upload ────────────────────────────────────────────────────── #
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing network traffic records",
        type=["csv"],
        help=(
            "Upload UNSW_NB15_testing-set.csv or any CSV with compatible "
            "network feature columns. 'label', 'id', and 'attack_cat' will "
            "be stripped before prediction — they are never used as model inputs."
        ),
    )

    if uploaded_file is None:
        st.info(
            "👆 Upload a CSV file to get started. "
            "You can use `data/UNSW_NB15_testing-set.csv` as a test file."
        )
        return

    # ── Read and validate ──────────────────────────────────────────────── #
    try:
        df_raw = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as exc:
        st.error(f"**Could not read the uploaded file:** {exc}")
        return

    if df_raw.empty:
        st.error("The uploaded file is empty.")
        return

    df_norm = _normalize_columns(df_raw.copy())

    st.success(
        f"✅ File uploaded: **{uploaded_file.name}** — "
        f"{len(df_norm):,} records, {df_norm.shape[1]} columns."
    )

    with st.expander("Preview uploaded data (first 5 rows)"):
        st.dataframe(df_norm.head())

    # ── Predict ────────────────────────────────────────────────────────── #
    with st.spinner("Running predictions …"):
        try:
            result_df = _run_prediction(pipeline, df_norm)
        except Exception as exc:
            st.error(f"**Prediction failed:** {exc}")
            logger.exception("Prediction error")
            return

    attack_count = int((result_df["predicted_label"] == 1).sum())
    normal_count = int((result_df["predicted_label"] == 0).sum())
    total = len(result_df)
    attack_pct = 100 * attack_count / total if total > 0 else 0

    # ── Summary metrics ────────────────────────────────────────────────── #
    st.divider()
    st.subheader("📊 Prediction Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{total:,}")
    with col2:
        st.metric("🟢 Normal", f"{normal_count:,}", delta=None)
    with col3:
        st.metric("🔴 Potential Attacks", f"{attack_count:,}", delta=f"{attack_pct:.1f}%")

    # ── Results table ──────────────────────────────────────────────────── #
    st.divider()
    st.subheader("📋 Prediction Results (first 200 rows)")

    preview_cols = ["predicted_label", "predicted_label_name", "attack_probability"]
    extra_cols = [
        c for c in result_df.columns
        if c not in preview_cols and c not in ("predicted_label",)
    ]
    display_cols = extra_cols[:6] + preview_cols  # show a few original cols + preds

    preview = result_df[display_cols].head(200).copy()

    def _colour_row(row):
        colour = "#ffcccc" if row["predicted_label"] == 1 else "#ccffcc"
        return [f"background-color: {colour}"] * len(row)

    styled = preview.style.apply(_colour_row, axis=1).format(
        {"attack_probability": "{:.4f}"}
    )
    st.dataframe(styled, use_container_width=True)

    # ── High-confidence attacks ────────────────────────────────────────── #
    high_conf = result_df[result_df["attack_probability"] >= 0.8]
    if not high_conf.empty:
        st.warning(
            f"⚠️ {len(high_conf):,} records have **attack probability ≥ 0.80** "
            "(high-confidence detections)."
        )

    # ── Download ───────────────────────────────────────────────────────── #
    st.divider()
    st.subheader("⬇️ Download Results")
    st.caption(
        "The downloaded CSV includes all original columns plus "
        "predicted_label, predicted_label_name, and attack_probability."
    )
    st.download_button(
        label="Download predictions as CSV",
        data=_df_to_csv_bytes(result_df),
        file_name="intrusion_detection_predictions.csv",
        mime="text/csv",
    )

    st.info(
        "**Note:** Attack probability scores are model estimates based on "
        "patterns in the UNSW-NB15 training dataset. They are not definitive "
        "indicators of malicious activity. Always combine model output with "
        "human expert analysis."
    )


if __name__ == "__main__":
    main()
