"""Evaluation utilities and visualizations for trained fraud model."""

from __future__ import annotations

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

from preprocessing import load_dataset, train_test_split_stratified


def save_roc_pr_curves(model, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=160)
    plt.close(fig)


def save_feature_importance(model, feature_names, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    estimator = model.named_steps["model"]
    importances = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])

    if importances is None:
        return

    order = np.argsort(importances)[::-1][:20]
    top_features = np.array(feature_names)[order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features[::-1], top_importances[::-1])
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=160)
    plt.close(fig)


def evaluate_model(data_path: str, model_path: str, output_dir: str) -> None:
    ds = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split_stratified(ds.X, ds.y)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out()

    save_roc_pr_curves(model, X_test, y_test, output_dir)
    save_feature_importance(model, feature_names, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained fraud model.")
    parser.add_argument("--data-path", default="data/creditcard.csv")
    parser.add_argument("--model-path", default="models/best_fraud_model.joblib")
    parser.add_argument("--output-dir", default="reports")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.data_path, args.model_path, args.output_dir)
