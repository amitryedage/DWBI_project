import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Design Palette (PRD Section 7)
PRIMARY_BLUE = "#1F4E9C"
CHURN_RED = "#D64545"
RETAINED_GREEN = "#2E9E6B"
CARD_BG = "#F4F6FA"
TEXT_COLOR = "#1B2430"
BORDER_COLOR = "#E2E8F0"

PLOT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="system-ui, -apple-system, sans-serif", color=TEXT_COLOR, size=12),
    margin=dict(l=40, r=30, t=50, b=40),
)


def create_donut_churn(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing Churned vs Retained proportions."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Churned vs Retained (No Data)")
        return fig

    counts = df["churned"].value_counts()
    churned_cnt = counts.get(1, 0)
    retained_cnt = counts.get(0, 0)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Retained", "Churned"],
                values=[retained_cnt, churned_cnt],
                hole=0.6,
                marker=dict(colors=[RETAINED_GREEN, CHURN_RED]),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textfont=dict(size=14),
            )
        ]
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Customer Retention vs Churn Ratio", font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def create_bar_churn_by_city(df: pd.DataFrame) -> go.Figure:
    """Bar chart of churn rate by city sorted by rate."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Churn Rate by City (No Data)")
        return fig

    city_data = (
        df.groupby("city", observed=False)
        .agg(
            churn_rate=("churned", lambda x: (x.mean() * 100.0)),
            customer_count=("customer_id", "count"),
        )
        .reset_index()
        .sort_values(by="churn_rate", ascending=True)
    )

    fig = go.Figure(
        go.Bar(
            x=city_data["churn_rate"],
            y=city_data["city"],
            orientation="h",
            marker=dict(color=PRIMARY_BLUE),
            text=city_data["churn_rate"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Churn Rate by City (%)", font=dict(size=16)),
        xaxis=dict(title="Churn Rate (%)", showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(title="", showgrid=False),
    )
    return fig


def create_bar_churn_by_balance_band(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing churn rate across balance bands."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Churn Rate by Balance Band (No Data)")
        return fig

    band_data = (
        df.groupby("balance_band", observed=False)
        .agg(
            churn_rate=("churned", lambda x: (x.mean() * 100.0) if len(x) > 0 else 0.0),
            total=("customer_id", "count"),
        )
        .reset_index()
    )

    fig = go.Figure(
        go.Bar(
            x=band_data["balance_band"].astype(str),
            y=band_data["churn_rate"],
            marker=dict(color=PRIMARY_BLUE),
            text=band_data["churn_rate"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>Balance Band: %{x}</b><br>Churn Rate: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Churn Rate by Balance Band", font=dict(size=16)),
        xaxis=dict(title="Balance Band", showgrid=False),
        yaxis=dict(title="Churn Rate (%)", showgrid=True, gridcolor="#EDF2F7"),
    )
    return fig


def create_balance_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of balance distribution grouped by churn status."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Balance Distribution (No Data)")
        return fig

    plot_df = df.copy()
    plot_df["Status"] = plot_df["churned"].map({1: "Churned", 0: "Retained"})

    fig = px.histogram(
        plot_df,
        x="balance",
        color="Status",
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={"Retained": RETAINED_GREEN, "Churned": CHURN_RED},
        nbins=40,
        labels={"balance": "Account Balance (INR)", "Status": "Customer Status"},
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Account Balance Distribution (Retained vs Churned)", font=dict(size=16)),
        xaxis=dict(title="Balance (INR)", showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(title="Customer Count", showgrid=True, gridcolor="#EDF2F7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_line_churn_by_tenure(df: pd.DataFrame) -> go.Figure:
    """Line chart showing churn rate progression over tenure years."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Churn Rate by Tenure (No Data)")
        return fig

    tenure_data = (
        df.groupby("tenure_years", observed=False)
        .agg(churn_rate=("churned", lambda x: (x.mean() * 100.0)))
        .reset_index()
    )

    fig = go.Figure(
        go.Scatter(
            x=tenure_data["tenure_years"],
            y=tenure_data["churn_rate"],
            mode="lines+markers+text",
            line=dict(color=PRIMARY_BLUE, width=3),
            marker=dict(size=8, color=CHURN_RED),
            text=tenure_data["churn_rate"].apply(lambda v: f"{v:.1f}%"),
            textposition="top center",
            hovertemplate="<b>Tenure: %{x} Years</b><br>Churn Rate: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Churn Rate by Tenure (Years)", font=dict(size=16)),
        xaxis=dict(title="Tenure (Years)", dtick=1, showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(title="Churn Rate (%)", showgrid=True, gridcolor="#EDF2F7"),
    )
    return fig


def create_bar_customers_by_tenure_band(df: pd.DataFrame) -> go.Figure:
    """Bar chart of customer counts across tenure bands."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Customer Count by Tenure Band (No Data)")
        return fig

    band_data = (
        df.groupby("tenure_band", observed=False)
        .agg(
            total_customers=("customer_id", "count"),
            churned_customers=("churned", "sum"),
        )
        .reset_index()
    )
    band_data["retained_customers"] = band_data["total_customers"] - band_data["churned_customers"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Retained",
            x=band_data["tenure_band"].astype(str),
            y=band_data["retained_customers"],
            marker_color=RETAINED_GREEN,
            hovertemplate="Retained: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Churned",
            x=band_data["tenure_band"].astype(str),
            y=band_data["churned_customers"],
            marker_color=CHURN_RED,
            hovertemplate="Churned: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="stack",
        title=dict(text="Customer Volume by Tenure Band", font=dict(size=16)),
        xaxis=dict(title="Tenure Band", showgrid=False),
        yaxis=dict(title="Customer Count", showgrid=True, gridcolor="#EDF2F7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_bar_churn_by_products(df: pd.DataFrame) -> go.Figure:
    """Bar chart of churn rate by number of products held."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Churn Rate by Number of Products (No Data)")
        return fig

    prod_data = (
        df.groupby("num_products", observed=False)
        .agg(
            churn_rate=("churned", lambda x: (x.mean() * 100.0)),
            count=("customer_id", "count"),
        )
        .reset_index()
    )

    colors = [
        RETAINED_GREEN if r < 20 else (PRIMARY_BLUE if r < 40 else CHURN_RED)
        for r in prod_data["churn_rate"]
    ]

    fig = go.Figure(
        go.Bar(
            x=prod_data["num_products"].astype(str) + " Products",
            y=prod_data["churn_rate"],
            marker=dict(color=colors),
            text=prod_data["churn_rate"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.2f}%<br>Customers: %{customdata}<extra></extra>",
            customdata=prod_data["count"],
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Churn Rate by Products Held", font=dict(size=16)),
        xaxis=dict(title="Products Held", showgrid=False),
        yaxis=dict(title="Churn Rate (%)", showgrid=True, gridcolor="#EDF2F7"),
    )
    return fig


def create_heatmap_products_vs_activity(df: pd.DataFrame) -> go.Figure:
    """Heatmap showing churn rate across products held vs active status."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Products vs Activity Heatmap (No Data)")
        return fig

    heatmap_df = (
        df.groupby(["num_products", "is_active_member"], observed=False)
        .agg(churn_rate=("churned", lambda x: (x.mean() * 100.0) if len(x) > 0 else 0.0))
        .reset_index()
    )

    pivot = heatmap_df.pivot(
        index="num_products",
        columns="is_active_member",
        values="churn_rate",
    ).fillna(0)

    # Rename column labels for clarity
    pivot.columns = ["Inactive Member", "Active Member"]
    pivot.index = [f"{i} Products" for i in pivot.index]

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale=[
            [0.0, "#EBF8FF"],
            [0.3, "#90CDF4"],
            [0.6, "#F6AD55"],
            [1.0, CHURN_RED],
        ],
        labels=dict(x="Membership Activity", y="Products Held", color="Churn Rate (%)"),
        aspect="auto",
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Churn Heatmap: Products Held vs Activity Status", font=dict(size=16)),
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title="", showgrid=False),
    )
    return fig


def create_bar_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing relative feature importance for churn prediction."""
    if importance_df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Feature Importance (No Data)")
        return fig

    sorted_df = importance_df.sort_values(by="importance_pct", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=sorted_df["importance_pct"],
            y=sorted_df["feature"],
            orientation="h",
            marker=dict(color=PRIMARY_BLUE),
            text=sorted_df["importance_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Relative Impact: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Key Predictive Churn Drivers (Feature Importance)", font=dict(size=16)),
        xaxis=dict(title="Relative Predictive Importance (%)", showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(title="", showgrid=False),
    )
    return fig

