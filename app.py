"""
Retail Banking Customer Churn Dashboard
Streamlit Application Entry Point
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from src.charts import (
    create_bar_churn_by_balance_band,
    create_bar_churn_by_city,
    create_bar_churn_by_products,
    create_bar_customers_by_tenure_band,
    create_balance_distribution,
    create_donut_churn,
    create_heatmap_products_vs_activity,
    create_line_churn_by_tenure,
)
from src.data import load_data
from src.kpis import (
    compute_kpis,
    generate_top_insights,
    get_high_risk_customers,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS (Light Theme - PRD Section 7 & 10)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Retail Banking Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #FFFFFF;
        color: #1B2430;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #F4F6FA;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #1F4E9C !important;
    }

    /* Alert / Callout Banners */
    .risk-banner {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-left: 5px solid #D64545;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 15px 0 20px 0;
        color: #991B1B;
    }
    .insight-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1F4E9C;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.95rem;
        color: #1E293B;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        font-size: 1.02rem;
        font-weight: 600;
        color: #64748B;
        padding: 10px 18px;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1F4E9C !important;
        border-bottom-color: #1F4E9C !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def format_inr(value: float) -> str:
    """Format numbers into human-readable INR (Thousands / Lakhs / Crores)."""
    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f} L"
    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.1f} k"
    return f"₹{value:,.0f}"


def apply_filters(df: pd.DataFrame, selections: Dict[str, Any]) -> pd.DataFrame:
    """Filter the master DataFrame based on user selections."""
    filtered = df.copy()

    if selections["cities"]:
        filtered = filtered[filtered["city"].isin(selections["cities"])]

    if selections["genders"]:
        filtered = filtered[filtered["gender"].isin(selections["genders"])]

    if selections["age_groups"]:
        filtered = filtered[filtered["age_group"].isin(selections["age_groups"])]

    if selections["segments"]:
        filtered = filtered[filtered["customer_segment"].isin(selections["segments"])]

    if selections["activity_status"] != "All":
        is_active = 1 if selections["activity_status"] == "Active" else 0
        filtered = filtered[filtered["is_active_member"] == is_active]

    tenure_min, tenure_max = selections["tenure_range"]
    filtered = filtered[
        (filtered["tenure_years"] >= tenure_min) & (filtered["tenure_years"] <= tenure_max)
    ]

    return filtered


# -----------------------------------------------------------------------------
# 3. Data Loading
# -----------------------------------------------------------------------------
try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 4. Sidebar Controls (Global Filters)
# -----------------------------------------------------------------------------
st.sidebar.title("🏦 Dashboard Filters")
st.sidebar.caption("Filter 10,000 retail banking accounts")

# Reset Filters mechanism
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = 0

if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.session_state.reset_trigger += 1
    st.rerun()

all_cities = sorted(df_raw["city"].unique().tolist())
all_genders = sorted(df_raw["gender"].unique().tolist())
all_age_groups = ["18-25", "26-35", "36-45", "46-55", "56+"]
all_segments = ["Mass", "Affluent", "Premium"]

selected_cities = st.sidebar.multiselect(
    "City",
    options=all_cities,
    default=all_cities,
    key=f"city_{st.session_state.reset_trigger}",
)

selected_genders = st.sidebar.multiselect(
    "Gender",
    options=all_genders,
    default=all_genders,
    key=f"gender_{st.session_state.reset_trigger}",
)

selected_age_groups = st.sidebar.multiselect(
    "Age Group",
    options=all_age_groups,
    default=all_age_groups,
    key=f"age_{st.session_state.reset_trigger}",
)

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=all_segments,
    default=all_segments,
    key=f"segment_{st.session_state.reset_trigger}",
)

activity_option = st.sidebar.radio(
    "Member Activity Status",
    options=["All", "Active", "Inactive"],
    index=0,
    horizontal=True,
    key=f"act_{st.session_state.reset_trigger}",
)

min_tenure = int(df_raw["tenure_years"].min())
max_tenure = int(df_raw["tenure_years"].max())

tenure_range = st.sidebar.slider(
    "Tenure (Years)",
    min_value=min_tenure,
    max_value=max_tenure,
    value=(min_tenure, max_tenure),
    step=1,
    key=f"tenure_{st.session_state.reset_trigger}",
)

# Apply filters
filter_params = {
    "cities": selected_cities,
    "genders": selected_genders,
    "age_groups": selected_age_groups,
    "segments": selected_segments,
    "activity_status": activity_option,
    "tenure_range": tenure_range,
}

df_filtered = apply_filters(df_raw, filter_params)
kpis = compute_kpis(df_filtered)


# -----------------------------------------------------------------------------
# 5. Header & Top KPI Banner
# -----------------------------------------------------------------------------
st.title("Retail Banking Customer Churn Dashboard")
st.markdown(
    "Comprehensive retention intelligence, capital risk analysis, and customer portfolio breakdown."
)

# Empty filter warning check
if df_filtered.empty:
    st.warning("⚠️ No customer records match the current filter criteria. Please broaden your sidebar filters.")
    st.stop()

# 5 Core KPI Metrics Row (PRD Section 5 & 6)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric(
        label="Churn Rate",
        value=f"{kpis['churn_rate']:.1f}%",
        delta=f"{kpis['churned_customers']:,} churned",
        delta_color="inverse",
    )
with c2:
    st.metric(
        label="Total Customers",
        value=f"{kpis['total_customers']:,}",
        delta=f"{kpis['retained_customers']:,} retained",
        delta_color="normal",
    )
with c3:
    st.metric(
        label="Avg Balance",
        value=format_inr(kpis["avg_balance"]),
    )
with c4:
    st.metric(
        label="Avg Tenure",
        value=f"{kpis['avg_tenure']:.1f} yrs",
    )
with c5:
    st.metric(
        label="Avg Products Held",
        value=f"{kpis['avg_products']:.2f}",
    )

st.markdown("<hr style='margin: 15px 0 25px 0; border: none; border-top: 1px solid #E2E8F0;' />", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. Tab Navigation (5 Core Pages)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Page 1: Overview",
    "💰 Page 2: Balance Analysis",
    "⏳ Page 3: Tenure Analysis",
    "📦 Page 4: Products Held",
    "⚠️ Page 5: Risk & Data",
])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("Executive Retention Overview")
    col_donut, col_city = st.columns([1, 1])

    with col_donut:
        fig_donut = create_donut_churn(df_filtered)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_city:
        fig_city = create_bar_churn_by_city(df_filtered)
        st.plotly_chart(fig_city, use_container_width=True)

    # Top Insights Section (Task 6.1)
    st.markdown("#### 💡 Key Portfolio Insights")
    insights = generate_top_insights(df_filtered)
    for insight in insights:
        st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)

# --- TAB 2: BALANCE ANALYSIS ---
with tab2:
    st.subheader("Balance Exposure & Churn Patterns")

    # Balance at Risk Banner
    bal_at_risk_str = format_inr(kpis["balance_at_risk"])
    st.markdown(
        f"""
        <div class='risk-banner'>
            <h4 style='margin:0 0 5px 0; color:#B91C1C;'>💰 Capital Exposure: Balance at Risk</h4>
            <span style='font-size:1.6rem; font-weight:800; color:#991B1B;'>{bal_at_risk_str}</span>
            <span style='font-size:0.95rem; margin-left:10px;'>has been lost or is actively at risk from churned customer accounts.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_band, col_dist = st.columns([1, 1])
    with col_band:
        fig_band = create_bar_churn_by_balance_band(df_filtered)
        st.plotly_chart(fig_band, use_container_width=True)

    with col_dist:
        fig_dist = create_balance_distribution(df_filtered)
        st.plotly_chart(fig_dist, use_container_width=True)

# --- TAB 3: TENURE ANALYSIS ---
with tab3:
    st.subheader("Customer Tenure & Loyalty Analysis")
    col_tenure_line, col_tenure_bar = st.columns([1, 1])

    with col_tenure_line:
        fig_line = create_line_churn_by_tenure(df_filtered)
        st.plotly_chart(fig_line, use_container_width=True)

    with col_tenure_bar:
        fig_bar = create_bar_customers_by_tenure_band(df_filtered)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 4: PRODUCTS HELD ---
with tab4:
    st.subheader("Product Cross-Selling & Retention Dynamics")
    col_prod_bar, col_heatmap = st.columns([1, 1])

    with col_prod_bar:
        fig_prod = create_bar_churn_by_products(df_filtered)
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_heatmap:
        fig_heat = create_heatmap_products_vs_activity(df_filtered)
        st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 5: RISK AND DATA ---
with tab5:
    st.subheader("Targeted High-Risk Customer Identification")
    st.markdown(
        """
        Customers flagged based on retention risk criteria:
        - **3 or more products held** (high drop-off risk)
        - **Inactive member with account balance > ₹1,00,000**
        - **Age 50+ with new tenure (≤ 2 years)**
        """
    )

    high_risk_df = get_high_risk_customers(df_filtered)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("High-Risk Accounts", f"{len(high_risk_df):,}")
    with r2:
        high_risk_bal = high_risk_df["balance"].sum() if not high_risk_df.empty else 0.0
        st.metric("Total High-Risk Balance", format_inr(high_risk_bal))
    with r3:
        high_risk_churn_rate = (
            (high_risk_df["churned"].mean() * 100.0) if not high_risk_df.empty else 0.0
        )
        st.metric("Historical Churn in Flagged Group", f"{high_risk_churn_rate:.1f}%")

    st.markdown("### High-Risk Customer Action List (Sorted by Balance)")
    display_cols = [
        "customer_id",
        "city",
        "gender",
        "age",
        "balance",
        "num_products",
        "is_active_member",
        "tenure_years",
        "customer_segment",
        "churned",
    ]

    st.dataframe(
        high_risk_df[display_cols].style.format(
            {
                "balance": "₹{:,.2f}",
                "is_active_member": lambda x: "Active" if x == 1 else "Inactive",
                "churned": lambda x: "Left" if x == 1 else "Active",
            }
        ),
        use_container_width=True,
        height=320,
    )

    # Download Buttons (Task 6.3)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        csv_filtered = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Current Filtered Data (CSV)",
            data=csv_filtered,
            file_name="filtered_banking_churn_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d_col2:
        csv_high_risk = high_risk_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⚠️ Download High-Risk Action List (CSV)",
            data=csv_high_risk,
            file_name="high_risk_customers_list.csv",
            mime="text/csv",
            use_container_width=True,
        )
