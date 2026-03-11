"""Streamlit app for fraud detection inference."""

from __future__ import annotations

import os

import joblib
import pandas as pd
import streamlit as st


DEFAULT_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource

def load_model(model_path: str):
    return joblib.load(model_path)


def get_feature_list(data_path: str):
    if os.path.exists(data_path):
        cols = pd.read_csv(data_path, nrows=1).columns.tolist()
        return [c for c in cols if c != "Class"]
    return DEFAULT_FEATURES


def main() -> None:
    st.set_page_config(page_title="Fraud Detection", layout="wide")
    st.title("💳 Fraud Detection System")

    model_path = st.sidebar.text_input("Model path", "models/best_fraud_model.joblib")
    data_path = st.sidebar.text_input("Dataset path (for columns)", "data/creditcard.csv")

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}. Train the model first.")
        st.stop()

    model = load_model(model_path)
    features = get_feature_list(data_path)

    st.write("Enter transaction values and click **Predict Fraud**.")

    cols = st.columns(4)
    transaction = {}
    for idx, feature in enumerate(features):
        with cols[idx % 4]:
            default_value = 0.0 if feature != "Amount" else 100.0
            transaction[feature] = st.number_input(feature, value=float(default_value), step=0.01)

    if st.button("Predict Fraud", type="primary"):
        input_df = pd.DataFrame([transaction])
        pred = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0, 1])

        st.metric("Fraud Probability", f"{proba:.4f}")
        if pred == 1:
            st.error("⚠️ Fraudulent transaction detected")
        else:
            st.success("✅ Legitimate transaction")

        st.write("Model output:")
        st.json({"prediction": pred, "fraud_probability": proba})


if __name__ == "__main__":
    main()
