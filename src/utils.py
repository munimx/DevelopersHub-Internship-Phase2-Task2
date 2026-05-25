import json
import logging
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from src import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the CSV in data/raw/ or pass --data-path."
        )
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in dataset.")
    y = df[config.TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Target column contains unexpected values outside Yes/No.")
    X = df.drop(columns=[config.TARGET_COLUMN, config.ID_COLUMN], errors="ignore")
    return X, y


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
