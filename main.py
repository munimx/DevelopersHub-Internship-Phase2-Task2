import argparse
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from src import config
from src.evaluate import compare_models, evaluate_model, save_reports
from src.train import train_models
from src.utils import get_logger, load_dataset, split_features_target

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate churn models.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Path to the Telco Churn CSV dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataset(args.data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    logistic_grid, rf_grid = train_models(X_train, y_train)

    models = {
        "Logistic Regression": logistic_grid.best_estimator_,
        "Random Forest": rf_grid.best_estimator_,
    }

    results = {
        name: evaluate_model(model, X_test, y_test) for name, model in models.items()
    }

    comparison = compare_models(results)
    LOGGER.info("Model comparison:\n%s", comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["Model"]
    best_model = models[best_model_name]

    joblib.dump(best_model, config.MODELS_DIR / "best_pipeline.joblib")
    LOGGER.info("Best model saved to models/best_pipeline.joblib")

    save_reports(
        results,
        comparison,
        best_model_name,
        best_model,
        X_test,
        y_test,
        df,
    )


if __name__ == "__main__":
    main()
