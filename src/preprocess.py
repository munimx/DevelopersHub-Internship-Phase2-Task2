from typing import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class NumericCoercer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Iterable[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "NumericCoercer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            return X
        data = X.copy()
        for column in self.columns:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        return data


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("coerce", NumericCoercer(numeric_features)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
