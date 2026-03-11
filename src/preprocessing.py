"""Preprocessing utilities for credit card fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


TARGET_COLUMN = "Class"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-inspired engineered features."""

    def __init__(self, amount_col: str = "Amount", time_col: str = "Time") -> None:
        self.amount_col = amount_col
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.amount_col in X.columns:
            X["LogAmount"] = np.log1p(X[self.amount_col].clip(lower=0))
            mad = (X[self.amount_col] - X[self.amount_col].median()).abs().median()
            X["AmountZ"] = (X[self.amount_col] - X[self.amount_col].median()) / (mad + 1e-6)
        if self.time_col in X.columns:
            seconds_in_day = 24 * 60 * 60
            X["Hour"] = (X[self.time_col] % seconds_in_day) / 3600
            X["DayCycleSin"] = np.sin(2 * np.pi * X["Hour"] / 24)
            X["DayCycleCos"] = np.cos(2 * np.pi * X["Hour"] / 24)
        return X


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


def load_dataset(csv_path: str, target_column: str = TARGET_COLUMN) -> DatasetBundle:
    """Load and minimally clean dataset."""
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates().reset_index(drop=True)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column].astype(int)

    return DatasetBundle(data=data, X=X, y=y)


def build_preprocessor() -> Pipeline:
    """Build preprocessing pipeline with imputation + scaling + feature engineering."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    col_transformer = ColumnTransformer(
        transformers=[("num", numeric_transformer, make_column_selector(dtype_include=np.number))],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor = Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("column_transformer", col_transformer),
        ]
    )
    return preprocessor


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
