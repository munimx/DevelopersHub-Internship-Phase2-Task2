import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import config
from src.utils import get_logger, load_dataset

LOGGER = get_logger(__name__)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[config.TARGET_COLUMN, config.ID_COLUMN], errors="ignore")


def run_inference(input_path: Path, model_path: Path, output_path: Path | None) -> None:
    data = load_dataset(input_path)
    model = joblib.load(model_path)
    features = prepare_features(data)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    output = pd.DataFrame(
        {
            "prediction": predictions,
            "probability": probabilities,
        }
    )
    if config.ID_COLUMN in data.columns:
        output.insert(0, config.ID_COLUMN, data[config.ID_COLUMN])

    if output_path is None:
        LOGGER.info("Inference completed. Showing first 5 rows:")
        LOGGER.info("\n%s", output.head())
        return

    output.to_csv(output_path, index=False)
    LOGGER.info("Predictions saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the best pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=config.MODELS_DIR / "best_pipeline.joblib",
        help="Path to the trained pipeline joblib file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save predictions as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(args.input, args.model, args.output)


if __name__ == "__main__":
    main()
