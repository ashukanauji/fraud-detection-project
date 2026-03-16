"""Preprocessing utilities for credit card fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import RobustScaler

TARGET_COLUMN = "Class"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-inspired engineered features."""

    def __init__(self, amount_col: str = "Amount", time_col: str = "Time") -> None:
        self.amount_col = amount_col
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
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
    """Load and safely clean dataset."""

    data = pd.read_csv(csv_path)

    # Replace infinite values with NaN
    data = data.replace([np.inf, -np.inf], np.nan)

    # Drop rows where target is NaN
    data = data.dropna(subset=[target_column])

    # Ensure target is numeric
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce")

    # Replace remaining NaN targets with 0
    data[target_column] = data[target_column].fillna(0)

    # Remove duplicates
    data = data.drop_duplicates().reset_index(drop=True)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = data.drop(columns=[target_column])

    # Convert target safely
    y = data[target_column].astype(int)

    return DatasetBundle(data=data, X=X, y=y)


def build_preprocessor():
    """Build preprocessing transformer."""

    numeric_transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                RobustScaler(),
                make_column_selector(dtype_include=np.number),
            )
        ],
        remainder="drop",
    )

    return numeric_transformer


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