from typing import Any, Dict, List
import pandas as pd


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute core banking KPIs (PRD Section 5).
    Handles empty DataFrames gracefully without raising errors.
    """
    total_customers = len(df)

    if total_customers == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "retained_customers": 0,
            "churn_rate": 0.0,
            "avg_balance": 0.0,
            "balance_at_risk": 0.0,
            "avg_tenure": 0.0,
            "avg_products": 0.0,
            "active_member_pct": 0.0,
            "credit_card_pct": 0.0,
            "avg_credit_score": 0.0,
        }

    churned_series = df["churned"]
    churned_customers = int(churned_series.sum())
    retained_customers = total_customers - churned_customers
    churn_rate = float(churned_series.mean() * 100.0)
    avg_balance = float(df["balance"].mean())
    balance_at_risk = float(df.loc[df["churned"] == 1, "balance"].sum())
    avg_tenure = float(df["tenure_years"].mean())
    avg_products = float(df["num_products"].mean())
    active_member_pct = float(df["is_active_member"].mean() * 100.0)
    credit_card_pct = float(df["has_credit_card"].mean() * 100.0)
    avg_credit_score = float(df["credit_score"].mean())

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "retained_customers": retained_customers,
        "churn_rate": churn_rate,
        "avg_balance": avg_balance,
        "balance_at_risk": balance_at_risk,
        "avg_tenure": avg_tenure,
        "avg_products": avg_products,
        "active_member_pct": active_member_pct,
        "credit_card_pct": credit_card_pct,
        "avg_credit_score": avg_credit_score,
    }


def churn_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate customer counts, churned count, churn rate, and balance at risk grouped by a column.
    """
    if df.empty or column not in df.columns:
        return pd.DataFrame(
            columns=[column, "customer_count", "churned_count", "churn_rate", "balance_at_risk", "avg_balance"]
        )

    df_temp = df.copy()
    df_temp["_risk_balance"] = df_temp["balance"].where(df_temp["churned"] == 1, 0.0)

    grouped = (
        df_temp.groupby(column, observed=False)
        .agg(
            customer_count=("churned", "count"),
            churned_count=("churned", "sum"),
            churn_rate=("churned", lambda x: (x.mean() * 100.0) if len(x) > 0 else 0.0),
            balance_at_risk=("_risk_balance", "sum"),
            avg_balance=("balance", "mean"),
        )
        .reset_index()
    )
    return grouped


def get_segment_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute customer volume, churn rate, and capital exposure per customer segment.
    """
    return churn_by(df, "customer_segment")


def get_high_risk_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter high-risk customers based on PRD rule (Task 6.2):
    - num_products >= 3
    OR (is_active_member == 0 AND balance > 100,000)
    OR (age >= 50 AND tenure_years <= 2)
    Sorted by balance descending.
    """
    if df.empty:
        return df.copy()

    condition = (
        (df["num_products"] >= 3)
        | ((df["is_active_member"] == 0) & (df["balance"] > 100000))
        | ((df["age"] >= 50) & (df["tenure_years"] <= 2))
    )

    high_risk_df = df[condition].copy()
    high_risk_df = high_risk_df.sort_values(by="balance", ascending=False)
    return high_risk_df


def generate_top_insights(df: pd.DataFrame) -> List[str]:
    """Generate dynamic data-driven insights based on current dataset/filter."""
    if df.empty:
        return ["No data available to generate insights."]

    insights: List[str] = []
    overall_churn = df["churned"].mean() * 100.0

    # Insight 1: Multi-product churn risk (3+ products)
    multi_prod = df[df["num_products"] >= 3]
    if len(multi_prod) > 0:
        multi_churn = multi_prod["churned"].mean() * 100.0
        insights.append(
            f"Customers with 3+ products show a churn rate of **{multi_churn:.1f}%** "
            f"(vs overall portfolio average of **{overall_churn:.1f}%**)."
        )

    # Insight 2: Inactive high-balance accounts
    inactive_high_bal = df[(df["is_active_member"] == 0) & (df["balance"] > 100000)]
    if len(inactive_high_bal) > 0:
        inact_churn = inactive_high_bal["churned"].mean() * 100.0
        total_inact_bal = inactive_high_bal["balance"].sum()
        insights.append(
            f"Inactive customers with balances over ₹1 Lakh have a **{inact_churn:.1f}%** churn rate, "
            f"representing ₹{total_inact_bal:,.0f} in capital at risk."
        )

    # Insight 3: City with highest churn rate
    city_churn = churn_by(df, "city")
    if not city_churn.empty:
        highest_city = city_churn.sort_values(by="churn_rate", ascending=False).iloc[0]
        insights.append(
            f"**{highest_city['city']}** exhibits the highest churn rate at **{highest_city['churn_rate']:.1f}%** "
            f"with {int(highest_city['churned_count']):,} churned customers."
        )

    return insights[:3]
