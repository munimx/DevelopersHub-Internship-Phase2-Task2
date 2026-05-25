# Telco Customer Churn — End-to-End ML Pipeline

## Problem Statement
Build a production-ready churn prediction system for Telco customers using scikit-learn pipelines, hyperparameter tuning, and reproducible evaluation.

## Dataset
The project uses the **Telco Customer Churn** dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV in `data/raw/` with the filename:
`WA_Fn-UseC_-Telco-Customer-Churn.csv`

Example download (requires Kaggle CLI configured):
```bash
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

## Pipeline Architecture
All preprocessing is performed inside the pipeline:

1. **Numeric Pipeline**
   - Invalid numeric coercion
   - Median imputation
   - Standard scaling
2. **Categorical Pipeline**
   - Most-frequent imputation
   - One-hot encoding (`handle_unknown="ignore"`)

These are composed with a `ColumnTransformer` and used in model pipelines.

## Model Training
Two models are trained using `GridSearchCV` with 5-fold CV:

- **Logistic Regression**
- **Random Forest**

Split strategy: **80/20** train-test split with stratification and `random_state=42`.

## Hyperparameter Search
Scoring metrics: **f1**, **accuracy**, and **roc_auc**, with **f1** as the refit metric.

## Results
Model performance is saved to:
```
reports/metrics.json
reports/classification_report.txt
```

A comparison table is logged during training with:
| Model | Accuracy | F1 | ROC-AUC |

## Export
Trained models are exported with `joblib`:
```
models/logistic_pipeline.joblib
models/rf_pipeline.joblib
models/best_pipeline.joblib
```

## Usage
Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train and evaluate:
```bash
python main.py
```

Run inference:
```bash
python -m src.predict --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output reports/predictions.csv
```

## Visualizations
Generated in `reports/`:
1. Confusion Matrix
2. ROC Curve
3. Feature Importance
4. Target Distribution
5. Correlation Heatmap
