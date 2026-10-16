"""
KPI Cross-Check and Validation Script (PRD Task 7.5).
Cross-checks 3 KPI values between manual raw calculation and module functions.
Run with: python scripts/verify_kpis.py
"""

from pathlib import Path
import pandas as pd
from src.data import load_data
from src.kpis import compute_kpis, churn_by, get_high_risk_customers


def run_crosscheck():
    print("=" * 65)
    print("TASK 7.5: KPI CROSS-CHECK & MATHEMATICAL AUDIT")
    print("=" * 65)

    df = load_data()
    module_kpis = compute_kpis(df)

    # ---------------------------------------------------------
    # Cross-Check 1: Overall Portfolio Churn Rate
    # ---------------------------------------------------------
    manual_churn_rate = (df["churned"].sum() / len(df)) * 100.0
    module_churn_rate = module_kpis["churn_rate"]
    diff_1 = abs(manual_churn_rate - module_churn_rate)

    print("\n[Cross-Check 1: Portfolio Churn Rate]")
    print(f"  • Manual Formula (sum/len * 100) : {manual_churn_rate:.4f}%")
    print(f"  • Module Output                  : {module_churn_rate:.4f}%")
    print(f"  • Absolute Discrepancy           : {diff_1:.6f}")
    assert diff_1 < 1e-5, "Mismatch on Churn Rate!"
    print("  -> Status: PASSED (Exact Match)")

    # ---------------------------------------------------------
    # Cross-Check 2: Aggregate Balance at Risk
    # ---------------------------------------------------------
    manual_bal_risk = df.loc[df["churned"] == 1, "balance"].sum()
    module_bal_risk = module_kpis["balance_at_risk"]
    diff_2 = abs(manual_bal_risk - module_bal_risk)

    print("\n[Cross-Check 2: Balance at Risk]")
    print(f"  • Manual Formula (sum of churned): ₹{manual_bal_risk:,.2f}")
    print(f"  • Module Output                  : ₹{module_bal_risk:,.2f}")
    print(f"  • Absolute Discrepancy           : ₹{diff_2:.4f}")
    assert diff_2 < 1e-4, "Mismatch on Balance at Risk!"
    print("  -> Status: PASSED (Exact Match)")

    # ---------------------------------------------------------
    # Cross-Check 3: Multi-Product (3+) Churn Rate
    # ---------------------------------------------------------
    multi_subset = df[df["num_products"] >= 3]
    manual_multi_churn = (multi_subset["churned"].mean()) * 100.0

    prod_grouped = churn_by(df, "num_products")
    # Weighted average of 3 and 4 products from churn_by
    p3_4 = prod_grouped[prod_grouped["num_products"] >= 3]
    module_multi_churn = (
        (p3_4["churned_count"].sum() / p3_4["customer_count"].sum()) * 100.0
    )
    diff_3 = abs(manual_multi_churn - module_multi_churn)

    print("\n[Cross-Check 3: Multi-Product (3+) Churn Rate]")
    print(f"  • Manual Subset Calculation      : {manual_multi_churn:.4f}%")
    print(f"  • GroupBy Re-aggregation         : {module_multi_churn:.4f}%")
    print(f"  • Absolute Discrepancy           : {diff_3:.6f}")
    assert diff_3 < 1e-5, "Mismatch on Multi-Product Churn Rate!"
    print("  -> Status: PASSED (Exact Match)")

    print("\n" + "=" * 65)
    print("ALL 3 KPI CROSS-CHECKS PASSED AUDIT SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    run_crosscheck()
