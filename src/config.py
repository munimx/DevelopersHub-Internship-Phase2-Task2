from pathlib import Path

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

RAW_DATA_FILENAME: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_PATH: Path = RAW_DATA_DIR / RAW_DATA_FILENAME

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

NUMERIC_FEATURES: list[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
