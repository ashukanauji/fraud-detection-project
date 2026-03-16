"""Train fraud detection models and select the best one."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocessing import build_preprocessor, load_dataset, train_test_split_stratified


RANDOM_STATE = 42


@dataclass
class ModelResult:
    model_name: str
    imbalance_strategy: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: List[List[int]]


def get_models(scale_pos_weight: float) -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight,
        ),
        "svm": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def get_imbalance_step(strategy: str):
    if strategy == "none":
        return None
    if strategy == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    raise ValueError(f"Unknown imbalance strategy: {strategy}")


def evaluate_predictions(y_true, y_pred, y_proba):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def train_all_models(data_path: str, model_output_path: str, metrics_output_path: str) -> Tuple[pd.DataFrame, str]:

    ds = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split_stratified(ds.X, ds.y)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = get_models(scale_pos_weight)

    results: List[ModelResult] = []

    best_f1 = -1.0
    best_pipeline = None
    best_name = ""

    for strategy in ["none", "smote", "undersample"]:

        for model_name, model in models.items():

            sampler = get_imbalance_step(strategy)

            steps = [
                ("preprocessor", build_preprocessor())
            ]

            if sampler:
                steps.append(("sampler", sampler))

            steps.append(("model", model))

            pipeline = ImbPipeline(steps=steps)

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            scores = evaluate_predictions(y_test, y_pred, y_proba)
            cm = confusion_matrix(y_test, y_pred)

            result = ModelResult(
                model_name=model_name,
                imbalance_strategy=strategy,
                precision=scores["precision"],
                recall=scores["recall"],
                f1=scores["f1"],
                roc_auc=scores["roc_auc"],
                pr_auc=scores["pr_auc"],
                confusion_matrix=cm.tolist(),
            )

            results.append(result)

            if result.f1 > best_f1:
                best_f1 = result.f1
                best_pipeline = pipeline
                best_name = f"{model_name} + {strategy}"

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    joblib.dump(best_pipeline, model_output_path)

    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values(
        by=["f1", "roc_auc"], ascending=False
    )

    metrics_df.to_csv(metrics_output_path, index=False)

    summary = {
        "best_model": best_name,
        "best_f1": float(best_f1),
        "model_path": model_output_path,
        "metrics_path": metrics_output_path,
    }

    with open(metrics_output_path.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return metrics_df, best_name


def parse_args():

    parser = argparse.ArgumentParser(description="Train fraud detection models.")

    parser.add_argument("--data-path", default="data/creditcard.csv")

    parser.add_argument("--model-output", default="models/best_fraud_model.joblib")

    parser.add_argument("--metrics-output", default="models/model_comparison.csv")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    results_df, best = train_all_models(
        args.data_path,
        args.model_output,
        args.metrics_output,
    )

    print(results_df.head(10).to_string(index=False))

    print(f"\nBest model: {best}")