from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src import config
from src.preprocess import NumericCoercer
from src.utils import ensure_dir, get_logger, save_json

LOGGER = get_logger(__name__)


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def get_feature_names(
    preprocessor, numeric_features: list[str], categorical_features: list[str]
) -> List[str]:
    num_features = list(numeric_features)
    cat_transformer = preprocessor.named_transformers_.get("cat")
    if cat_transformer is None:
        return num_features
    ohe = cat_transformer.named_steps["onehot"]
    cat_features = list(ohe.get_feature_names_out(categorical_features))
    return num_features + cat_features


def plot_confusion(cm: pd.DataFrame, path: str) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    ensure_dir(config.REPORTS_DIR)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc(y_true: pd.Series, y_prob: pd.Series, path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    ensure_dir(config.REPORTS_DIR)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_target_distribution(y: pd.Series, path: str) -> None:
    counts = y.value_counts().rename({0: "No", 1: "Yes"})
    plt.figure(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title("Target Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    ensure_dir(config.REPORTS_DIR)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, path: str) -> None:
    numeric_df = df[config.NUMERIC_FEATURES].copy()
    numeric_df = NumericCoercer(config.NUMERIC_FEATURES).transform(numeric_df)
    corr = numeric_df.corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    ensure_dir(config.REPORTS_DIR)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_feature_importance(
    model: Pipeline, feature_names: List[str], path: str
) -> None:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = abs(estimator.coef_).ravel()
    else:
        LOGGER.warning("Model does not support feature importance.")
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="mako")
    plt.title("Top Feature Importances")
    ensure_dir(config.REPORTS_DIR)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = result["metrics"]
        rows.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "F1": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            }
        )
    return pd.DataFrame(rows).sort_values(by="F1", ascending=False)


def save_reports(
    results: Dict[str, Dict],
    comparison: pd.DataFrame,
    best_model_name: str,
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_df: pd.DataFrame,
) -> None:
    ensure_dir(config.REPORTS_DIR)

    metrics_payload = {
        "comparison_table": comparison.to_dict(orient="records"),
        "best_model": best_model_name,
        "metrics": {name: result["metrics"] for name, result in results.items()},
    }
    save_json(metrics_payload, config.REPORTS_DIR / "metrics.json")

    best_result = results[best_model_name]
    report_path = config.REPORTS_DIR / "classification_report.txt"
    report_path.write_text(best_result["report"], encoding="utf-8")

    plot_confusion(
        best_result["confusion_matrix"],
        str(config.REPORTS_DIR / "confusion_matrix.png"),
    )
    plot_roc(y_test, best_result["y_prob"], str(config.REPORTS_DIR / "roc_curve.png"))
    plot_target_distribution(y_test, str(config.REPORTS_DIR / "target_distribution.png"))
    plot_correlation_heatmap(raw_df, str(config.REPORTS_DIR / "correlation_heatmap.png"))

    feature_names = get_feature_names(
        best_model.named_steps["preprocess"],
        config.NUMERIC_FEATURES,
        config.CATEGORICAL_FEATURES,
    )
    plot_feature_importance(
        best_model, feature_names, str(config.REPORTS_DIR / "feature_importance.png")
    )
