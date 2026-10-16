

from typing import Any, Dict
import pandas as pd


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
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
        }

    churned_series = df["churned"]
    churned_customers = int(churned_series.sum())
    retained_customers = total_customers - churned_customers
    churn_rate = float(churned_series.mean() * 100.0)
    avg_balance = float(df["balance"].mean())
    balance_at_risk = float(df.loc[df["churned"] == 1, "balance"].sum())
    avg_tenure = float(df["tenure_years"].mean())
    avg_products = float(df["num_products"].mean())

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "retained_customers": retained_customers,
        "churn_rate": churn_rate,
        "avg_balance": avg_balance,
        "balance_at_risk": balance_at_risk,
        "avg_tenure": avg_tenure,
        "avg_products": avg_products,
    }


def churn_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "customer_count", "churned_count", "churn_rate"])

    grouped = (
        df.groupby(column, observed=False)
        .agg(
            customer_count=("churned", "count"),
            churned_count=("churned", "sum"),
            churn_rate=("churned", lambda x: (x.mean() * 100.0) if len(x) > 0 else 0.0),
        )
        .reset_index()
    )
    return grouped


def get_high_risk_customers(df: pd.DataFrame) -> pd.DataFrame:
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


def generate_top_insights(df: pd.DataFrame) -> list[str]:
    """Generate dynamic data-driven insights based on current dataset/filter."""
    if df.empty:
        return ["No data available to generate insights."]

    insights = []
    overall_churn = df["churned"].mean() * 100.0

    # Insight 1: Multi-product churn risk (3+ products)
    multi_prod = df[df["num_products"] >= 3]
    if len(multi_prod) > 0:
        multi_churn = multi_prod["churned"].mean() * 100.0
        insights.append(
            f"Customers with 3+ products show a churn rate of **{multi_churn:.1f}%** "
            f"(vs overall average of **{overall_churn:.1f}%**)."
        )

    # Insight 2: Inactive high-balance accounts
    inactive_high_bal = df[(df["is_active_member"] == 0) & (df["balance"] > 100000)]
    if len(inactive_high_bal) > 0:
        inact_churn = inactive_high_bal["churned"].mean() * 100.0
        total_inact_bal = inactive_high_bal["balance"].sum()
        insights.append(
            f"Inactive customers with balances over ₹1 Lakh have a **{inact_churn:.1f}%** churn rate, "
            f"accounting for ₹{total_inact_bal:,.0f} in capital."
        )

    # Insight 3: City with highest churn rate
    city_churn = churn_by(df, "city")
    if not city_churn.empty:
        highest_city = city_churn.sort_values(by="churn_rate", ascending=False).iloc[0]
        insights.append(
            f"**{highest_city['city']}** has the highest churn rate at **{highest_city['churn_rate']:.1f}%** "
            f"with {int(highest_city['churned_count']):,} churned customers."
        )

    return insights[:3]
