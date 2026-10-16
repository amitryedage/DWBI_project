

from pathlib import Path
import shutil
import pandas as pd
import streamlit as st

EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "city",
    "customer_segment",
    "tenure_years",
    "balance",
    "num_products",
    "has_credit_card",
    "is_active_member",
    "credit_score",
    "estimated_salary",
    "churned",
]

AGE_BINS = [17, 25, 35, 45, 55, 120]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56+"]

TENURE_BINS = [-1, 2, 5, 8, 10]
TENURE_LABELS = ["0-2", "3-5", "6-8", "9-10"]

BALANCE_LABELS = ["Zero", "1 to 50k", "50k to 100k", "100k to 150k", "150k+"]


def resolve_data_path() -> Path:
    """Find or setup the dataset location in data/ or root directory."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    data_csv = data_dir / "retail_banking_churn.csv"
    root_csv = root_dir / "retail_banking_churn.csv"

    # If data/retail_banking_churn.csv exists, use it
    if data_csv.is_file():
        return data_csv

    # If root_csv exists, copy it to data/ if possible, or return root_csv
    if root_csv.is_file():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(root_csv, data_csv)
            return data_csv
        except Exception:
            return root_csv

    raise FileNotFoundError(
        f"Dataset not found. Checked {data_csv} and {root_csv}."
    )


def validate_data(df: pd.DataFrame) -> None:
    """Validate schema, nulls, and value ranges."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    if df.empty:
        raise ValueError("The dataset is empty.")

    unique_churn = set(df["churned"].dropna().unique())
    if not unique_churn.issubset({0, 1}):
        raise ValueError(f"Unexpected values in 'churned' column: {unique_churn}")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create age_group, balance_band, and tenure_band derived columns."""
    df = df.copy()

    # Derived Age Group
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
    ).astype(str)

    # Derived Tenure Band
    df["tenure_band"] = pd.cut(
        df["tenure_years"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
    ).astype(str)

    # Derived Balance Band
    def categorize_balance(b: float) -> str:
        if b <= 0.0:
            return "Zero"
        if b <= 50000:
            return "1 to 50k"
        if b <= 100000:
            return "50k to 100k"
        if b <= 150000:
            return "100k to 150k"
        return "150k+"

    df["balance_band"] = df["balance"].apply(categorize_balance)

    # Set as categorical with fixed order for consistent sorting in charts
    df["age_group"] = pd.Categorical(df["age_group"], categories=AGE_LABELS, ordered=True)
    df["tenure_band"] = pd.Categorical(df["tenure_band"], categories=TENURE_LABELS, ordered=True)
    df["balance_band"] = pd.Categorical(df["balance_band"], categories=BALANCE_LABELS, ordered=True)

    return df


@st.cache_data
def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load, validate, and enrich the retail banking churn dataset."""
    if filepath is None:
        target_path = resolve_data_path()
    else:
        target_path = Path(filepath)

    df = pd.read_csv(target_path)
    validate_data(df)
    df = add_derived_columns(df)
    return df
