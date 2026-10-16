"""
Unit tests for data loading, validation, KPI calculations, and risk identification.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data import add_derived_columns, load_data, validate_data
from src.kpis import churn_by, compute_kpis, get_high_risk_customers


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture providing a deterministic small dataset for unit testing."""
    data = {
        "customer_id": [1, 2, 3, 4, 5],
        "age": [22, 35, 48, 55, 62],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "city": ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai"],
        "customer_segment": ["Mass", "Affluent", "Mass", "Premium", "Affluent"],
        "tenure_years": [1, 4, 7, 2, 10],
        "balance": [0.0, 75000.0, 120000.0, 200000.0, 50000.0],
        "num_products": [1, 2, 3, 1, 4],
        "has_credit_card": [1, 1, 0, 1, 1],
        "is_active_member": [1, 0, 0, 0, 1],
        "credit_score": [650, 700, 580, 750, 690],
        "estimated_salary": [50000.0, 80000.0, 60000.0, 120000.0, 95000.0],
        "churned": [0, 0, 1, 1, 0],
    }
    df = pd.DataFrame(data)
    return add_derived_columns(df)


def test_derived_columns(sample_df: pd.DataFrame):
    """Verify age_group, balance_band, and tenure_band creation."""
    assert "age_group" in sample_df.columns
    assert "balance_band" in sample_df.columns
    assert "tenure_band" in sample_df.columns

    # Check zero balance categorization
    assert sample_df.loc[sample_df["balance"] == 0.0, "balance_band"].iloc[0] == "Zero"
    # Check 120k balance categorization
    assert sample_df.loc[sample_df["balance"] == 120000.0, "balance_band"].iloc[0] == "100k to 150k"


def test_compute_kpis(sample_df: pd.DataFrame):
    """Test standard KPI calculations."""
    kpis = compute_kpis(sample_df)

    assert kpis["total_customers"] == 5
    assert kpis["churned_customers"] == 2
    assert kpis["retained_customers"] == 3
    assert kpis["churn_rate"] == 40.0
    assert kpis["avg_balance"] == (0 + 75000 + 120000 + 200000 + 50000) / 5
    assert kpis["balance_at_risk"] == 120000.0 + 200000.0
    assert kpis["avg_tenure"] == (1 + 4 + 7 + 2 + 10) / 5
    assert kpis["avg_products"] == (1 + 2 + 3 + 1 + 4) / 5


def test_compute_kpis_empty():
    """Test that empty data returns zeros without errors."""
    empty_df = pd.DataFrame(columns=["customer_id", "churned", "balance", "tenure_years", "num_products"])
    kpis = compute_kpis(empty_df)

    assert kpis["total_customers"] == 0
    assert kpis["churn_rate"] == 0.0
    assert kpis["balance_at_risk"] == 0.0
    assert kpis["avg_balance"] == 0.0


def test_churn_by_helper(sample_df: pd.DataFrame):
    """Test grouping by city and balance at risk."""
    grouped = churn_by(sample_df, "city")
    assert "city" in grouped.columns
    assert "customer_count" in grouped.columns
    assert "churn_rate" in grouped.columns
    assert "balance_at_risk" in grouped.columns

    pune_row = grouped[grouped["city"] == "Pune"].iloc[0]
    assert pune_row["customer_count"] == 2
    assert pune_row["churn_rate"] == 50.0  # 1 out of 2 churned
    assert pune_row["balance_at_risk"] == 120000.0  # customer 3 churned with 120k


def test_segment_kpis(sample_df: pd.DataFrame):
    """Test segment KPI aggregation."""
    from src.kpis import get_segment_kpis
    seg_df = get_segment_kpis(sample_df)
    assert "customer_segment" in seg_df.columns
    assert len(seg_df) > 0


def test_high_risk_rule(sample_df: pd.DataFrame):
    """
    High-risk customers:
    - num_products >= 3
    OR (is_active_member == 0 and balance > 100k)
    OR (age >= 50 and tenure_years <= 2)
    """
    high_risk = get_high_risk_customers(sample_df)
    flagged_ids = set(high_risk["customer_id"].tolist())

    # Customer 3: num_products = 3, inactive, balance 120k -> flagged
    # Customer 4: age = 55, tenure = 2 -> flagged
    # Customer 5: num_products = 4 -> flagged
    assert 3 in flagged_ids
    assert 4 in flagged_ids
    assert 5 in flagged_ids
    assert 1 not in flagged_ids


def test_full_dataset_churn_rate_acceptance_criterion():
    """Verify that full 10,000 dataset loads and churn rate is ~20% (Acceptance Criterion 5)."""
    root_csv = Path("retail_banking_churn.csv")
    data_csv = Path("data/retail_banking_churn.csv")

    target = root_csv if root_csv.exists() else data_csv
    if target.exists():
        df = pd.read_csv(target)
        validate_data(df)
        assert len(df) == 10000
        churn_rate = df["churned"].mean() * 100.0
        # Overall churn rate should be approximately 20% (+/- 2%)
        assert 18.0 <= churn_rate <= 22.0


def test_kpi_crosscheck_task_7_5():
    """Cross-check 3 KPI values manually against pandas ground truth (Task 7.5)."""
    root_csv = Path("retail_banking_churn.csv")
    data_csv = Path("data/retail_banking_churn.csv")
    target = root_csv if root_csv.exists() else data_csv

    if target.exists():
        df = pd.read_csv(target)
        kpis = compute_kpis(df)

        # Cross-check 1: Churn Rate
        manual_churn = (df["churned"].sum() / len(df)) * 100.0
        assert abs(manual_churn - kpis["churn_rate"]) < 1e-5

        # Cross-check 2: Balance at Risk
        manual_bal_risk = df.loc[df["churned"] == 1, "balance"].sum()
        assert abs(manual_bal_risk - kpis["balance_at_risk"]) < 1e-4

        # Cross-check 3: Multi-product churn
        multi_subset = df[df["num_products"] >= 3]
        manual_multi = multi_subset["churned"].mean() * 100.0
        by_prod = churn_by(df, "num_products")
        p3_4 = by_prod[by_prod["num_products"] >= 3]
        module_multi = (p3_4["churned_count"].sum() / p3_4["customer_count"].sum()) * 100.0
        assert abs(manual_multi - module_multi) < 1e-5

