"""Inference helper for fraud detection model."""

from __future__ import annotations

import argparse
from typing import Dict

import joblib
import pandas as pd


class FraudPredictor:
    def __init__(self, model_path: str = "models/best_fraud_model.joblib") -> None:
        self.model = joblib.load(model_path)

    def predict_transaction(self, transaction: Dict[str, float]) -> Dict[str, float]:
        df = pd.DataFrame([transaction])
        pred = int(self.model.predict(df)[0])
        proba = float(self.model.predict_proba(df)[0, 1])
        return {
            "prediction": pred,
            "label": "Fraud" if pred == 1 else "Legitimate",
            "fraud_probability": proba,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-transaction prediction.")
    parser.add_argument("--model-path", default="models/best_fraud_model.joblib")
    parser.add_argument("--input-csv", required=True, help="CSV with a single transaction row.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = FraudPredictor(args.model_path)
    sample = pd.read_csv(args.input_csv).iloc[0].to_dict()
    result = predictor.predict_transaction(sample)
    print(result)
