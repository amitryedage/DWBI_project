"""
Quick Exploratory Data Analysis (EDA) Script (Task 2.4).
Run with: python scripts/eda.py
"""

from pathlib import Path
import pandas as pd
from src.data import add_derived_columns, load_data, validate_data
from src.kpis import compute_kpis, churn_by, get_high_risk_customers


def run_eda():
    print("=" * 60)
    print("RETAIL BANKING CHURN DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    print(f"\n1. DATASET SHAPE: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Schema and Data Types
    print("\n2. DATA TYPES & NULL COUNTS:")
    info_df = pd.DataFrame({
        "DataType": df.dtypes.astype(str),
        "NullCount": df.isnull().sum(),
        "UniqueValues": df.nunique(),
    })
    print(info_df.to_string())

    # 3. Overall KPIs
    kpis = compute_kpis(df)
    print("\n3. CORE PORTFOLIO KPIS:")
    print(f"   • Total Customers   : {kpis['total_customers']:,}")
    print(f"   • Churned Customers : {kpis['churned_customers']:,}")
    print(f"   • Retained Customers: {kpis['retained_customers']:,}")
    print(f"   • Overall Churn Rate: {kpis['churn_rate']:.2f}%")
    print(f"   • Average Balance   : ₹{kpis['avg_balance']:,.2f}")
    print(f"   • Balance at Risk   : ₹{kpis['balance_at_risk']:,.2f}")
    print(f"   • Average Tenure    : {kpis['avg_tenure']:.1f} years")
    print(f"   • Avg Products Held : {kpis['avg_products']:.2f}")

    # 4. Churn Breakdown by City
    print("\n4. CHURN BY CITY:")
    city_churn = churn_by(df, "city").sort_values(by="churn_rate", ascending=False)
    print(city_churn.to_string(index=False))

    # 5. Churn Breakdown by Number of Products
    print("\n5. CHURN BY NUMBER OF PRODUCTS:")
    prod_churn = churn_by(df, "num_products")
    print(prod_churn.to_string(index=False))

    # 6. Churn Breakdown by Balance Band
    print("\n6. CHURN BY BALANCE BAND:")
    band_churn = churn_by(df, "balance_band")
    print(band_churn.to_string(index=False))

    # 7. High-Risk Customer Identification (PRD 6.2)
    high_risk_df = get_high_risk_customers(df)
    high_risk_churn = high_risk_df["churned"].mean() * 100.0
    print(f"\n7. HIGH-RISK CUSTOMER COHORT:")
    print(f"   • Flagged Accounts  : {len(high_risk_df):,} ({len(high_risk_df)/len(df)*100:.1f}% of total)")
    print(f"   • Historical Churn  : {high_risk_churn:.2f}%")
    print(f"   • Total Bal at Risk : ₹{high_risk_df['balance'].sum():,.2f}")
    print("=" * 60)
    print("EDA completed successfully.")


if __name__ == "__main__":
    run_eda()
