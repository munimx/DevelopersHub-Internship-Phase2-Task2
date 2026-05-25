from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src import config
from src.preprocess import build_preprocessor
from src.utils import ensure_dir, get_logger

LOGGER = get_logger(__name__)


def build_logistic_pipeline(preprocessor) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    random_state=config.RANDOM_STATE,
                    max_iter=1000,
                ),
            ),
        ]
    )


def build_rf_pipeline(preprocessor) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def tune_model(
    pipeline: Pipeline, param_grid: Dict[str, list], X_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    scoring = {"f1": "f1", "accuracy": "accuracy", "roc_auc": "roc_auc"}
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=config.CV_FOLDS,
        refit="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[GridSearchCV, GridSearchCV]:
    preprocessor = build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES)

    LOGGER.info("Training Logistic Regression with GridSearchCV")
    logistic_pipeline = build_logistic_pipeline(preprocessor)
    logistic_params = {"model__C": [0.01, 0.1, 1, 10]}
    logistic_grid = tune_model(logistic_pipeline, logistic_params, X_train, y_train)

    LOGGER.info("Training Random Forest with GridSearchCV")
    rf_pipeline = build_rf_pipeline(preprocessor)
    rf_params = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5, 10],
    }
    rf_grid = tune_model(rf_pipeline, rf_params, X_train, y_train)

    ensure_dir(config.MODELS_DIR)
    joblib.dump(logistic_grid.best_estimator_, config.MODELS_DIR / "logistic_pipeline.joblib")
    joblib.dump(rf_grid.best_estimator_, config.MODELS_DIR / "rf_pipeline.joblib")

    LOGGER.info("Saved tuned pipelines to models/")
    return logistic_grid, rf_grid
