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
    create_bar_feature_importance,
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
    get_segment_kpis,
)
from src.model import predict_single_customer_churn, train_churn_model

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

    # If user deselected all options in any multiselect, return empty DataFrame cleanly
    if not selections["cities"] or not selections["genders"] or not selections["age_groups"] or not selections["segments"]:
        return filtered.iloc[0:0]

    filtered = filtered[filtered["city"].isin(selections["cities"])]
    filtered = filtered[filtered["gender"].isin(selections["genders"])]
    filtered = filtered[filtered["age_group"].isin(selections["age_groups"])]
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    " Page 1: Overview",
    "Page 2: Balance Analysis",
    " Page 3: Tenure Analysis",
    " Page 4: Products Held",
    " Page 5: Risk & Data",
    "Page 6: What-If Simulator & ML",
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

    # Segment Performance Cards
    st.markdown("#### 👥 Customer Segment Retention & Risk")
    seg_df = get_segment_kpis(df_filtered)
    if not seg_df.empty:
        col_s1, col_s2, col_s3 = st.columns(3)
        cols_list = [col_s1, col_s2, col_s3]
        for idx, row in seg_df.iterrows():
            with cols_list[idx % 3]:
                st.markdown(
                    f"""
                    <div style='background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 16px; margin-bottom:10px;'>
                        <div style='font-size:0.82rem; font-weight:600; color:#64748B; text-transform:uppercase;'>{row['customer_segment']} Segment</div>
                        <div style='font-size:1.35rem; font-weight:700; color:#1F4E9C; margin:3px 0;'>{row['churn_rate']:.1f}% Churn</div>
                        <div style='font-size:0.85rem; color:#475569;'>{int(row['customer_count']):,} accounts · {format_inr(row['balance_at_risk'])} at risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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

    # Actionable Filter Controls for Risk Table
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        risk_filter_type = st.selectbox(
            "Filter by Specific Risk Factor:",
            options=["All Flagged Accounts", "Multi-Product (3+)", "Inactive High-Balance", "New Senior Account"],
            index=0,
        )
    with f_col2:
        search_query = st.text_input(
            "🔍 Search Account by Customer ID:",
            placeholder="e.g. 100005",
        )

    # Apply table filters
    action_table_df = high_risk_df.copy()
    if risk_filter_type != "All Flagged Accounts":
        action_table_df = action_table_df[
            action_table_df["risk_factors"].str.contains(risk_filter_type, na=False)
        ]

    if search_query.strip():
        action_table_df = action_table_df[
            action_table_df["customer_id"].astype(str).str.contains(search_query.strip(), na=False)
        ]

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
        "risk_factors",
        "churned",
    ]

    st.dataframe(
        action_table_df[display_cols].style.format(
            {
                "balance": "₹{:,.2f}",
                "is_active_member": lambda x: "Active" if x == 1 else "Inactive",
                "churned": lambda x: "Left Bank" if x == 1 else "Active",
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
            label=" Download Current Filtered Data (CSV)",
            data=csv_filtered,
            file_name="filtered_banking_churn_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d_col2:
        csv_high_risk = high_risk_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download High-Risk Action List (CSV)",
            data=csv_high_risk,
            file_name="high_risk_customers_list.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --- TAB 6: WHAT-IF SIMULATOR & ML (PRD STRETCH GOALS) ---
with tab6:
    st.subheader("Predictive Churn Modeling & 'What-If' Retention Simulator")
    st.markdown(
        "Leverage machine learning algorithms to evaluate customer attrition probability and simulate proactive retention interventions."
    )

    # 1. Model Configuration
    m_col1, m_col2 = st.columns([1, 1])
    with m_col1:
        chosen_algo = st.radio(
            "Predictive Algorithm Architecture:",
            options=["Random Forest Classifier", "Logistic Regression"],
            horizontal=True,
        )

    algo_name = "Logistic Regression" if "Logistic" in chosen_algo else "Random Forest"
    model_bundle = train_churn_model(df_raw, model_type=algo_name)

    # Model Metric Highlights
    mod_c1, mod_c2, mod_c3 = st.columns(3)
    with mod_c1:
        st.metric("Model Training Fit / Accuracy", f"{model_bundle['accuracy'] * 100:.1f}%")
    with mod_c2:
        engine_label = "Scikit-Learn Optimized" if model_bundle["sklearn_used"] else "NumPy Pure Engine"
        st.metric("Computation Engine", engine_label)
    with mod_c3:
        st.metric("Training Population", f"{len(df_raw):,} Customers")

    # 2. Feature Importance Chart
    st.markdown("#### 🧠 Predictive Feature Importance")
    fig_importance = create_bar_feature_importance(model_bundle["importance_df"])
    st.plotly_chart(fig_importance, use_container_width=True)

    # 3. Interactive "What-If" Simulator
    st.markdown("---")
    st.markdown("### 🕹️ Interactive Customer 'What-If' Retention Simulator")
    st.markdown(
        "Simulate intervention strategies on an account to quantify the expected reduction in churn probability."
    )

    sim_c1, sim_c2 = st.columns([1, 1])

    with sim_c1:
        st.markdown("##### 👤 Customer Portfolio Attributes")
        sim_num_prod = st.slider("Products Held", min_value=1, max_value=4, value=3)
        sim_active_str = st.selectbox("Member Activity Status", ["Inactive Member", "Active Member"], index=0)
        sim_is_active = 1 if "Active" in sim_active_str else 0
        sim_age = st.slider("Customer Age", min_value=18, max_value=80, value=52)
        sim_balance = st.number_input("Account Balance (INR)", min_value=0.0, max_value=300000.0, value=115000.0, step=5000.0)
        sim_tenure = st.slider("Tenure with Bank (Years)", min_value=0, max_value=10, value=1)
        sim_credit = st.slider("Credit Score", min_value=350, max_value=850, value=670)
        sim_card_str = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0)
        sim_card = 1 if sim_card_str == "Yes" else 0
        sim_salary = st.number_input("Estimated Salary (INR)", min_value=10000.0, max_value=250000.0, value=75000.0, step=5000.0)

    # Current profile dictionary
    customer_profile = {
        "age": sim_age,
        "balance": sim_balance,
        "num_products": sim_num_prod,
        "is_active_member": sim_is_active,
        "credit_score": sim_credit,
        "tenure_years": sim_tenure,
        "has_credit_card": sim_card,
        "estimated_salary": sim_salary,
    }

    predicted_prob = predict_single_customer_churn(model_bundle, customer_profile)

    with sim_c2:
        st.markdown("##### 📊 Real-Time Predicted Attrition Risk")

        # Color-coded risk status
        if predicted_prob >= 50.0:
            risk_color = "#D64545"
            risk_badge = "CRITICAL / HIGH RISK"
            badge_bg = "#FEF2F2"
        elif predicted_prob >= 25.0:
            risk_color = "#DD6B20"
            risk_badge = "MODERATE RISK"
            badge_bg = "#FFFAF0"
        else:
            risk_color = "#2E9E6B"
            risk_badge = "LOW RISK / SAFE"
            badge_bg = "#F0FDF4"

        st.markdown(
            f"""
            <div style='background-color:{badge_bg}; border:2px solid {risk_color}; border-radius:12px; padding:20px; text-align:center; margin-bottom:20px;'>
                <div style='font-size:0.9rem; font-weight:700; color:{risk_color}; letter-spacing:1px;'>{risk_badge}</div>
                <div style='font-size:3rem; font-weight:800; color:{risk_color}; margin:8px 0;'>{predicted_prob:.1f}%</div>
                <div style='font-size:0.95rem; color:#475569;'>Probability of this customer leaving the bank</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🎯 Simulated Retention Interventions")

        # Scenario 1: Convert to Active Member
        if sim_is_active == 0:
            active_profile = dict(customer_profile, is_active_member=1)
            active_prob = predict_single_customer_churn(model_bundle, active_profile)
            diff_active = predicted_prob - active_prob
            st.markdown(
                f"""
                <div class='insight-card'>
                    <b>Intervention 1: Re-engage as Active Member</b><br>
                    Predicted Churn drops from <b>{predicted_prob:.1f}%</b> to <b>{active_prob:.1f}%</b> 
                    (<span style='color:#2E9E6B; font-weight:700;'>-{diff_active:.1f}% reduction</span>).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='insight-card'>Customer is already an <b>Active Member</b>. Maintaining regular digital/branch engagement keeps attrition low.</div>",
                unsafe_allow_html=True,
            )

        # Scenario 2: Rationalize Products
        if sim_num_prod >= 3:
            opt_prod_profile = dict(customer_profile, num_products=2)
            opt_prod_prob = predict_single_customer_churn(model_bundle, opt_prod_profile)
            diff_prod = predicted_prob - opt_prod_prob
            st.markdown(
                f"""
                <div class='insight-card'>
                    <b>Intervention 2: Consolidate / Restructure Products to 2</b><br>
                    Predicted Churn drops from <b>{predicted_prob:.1f}%</b> to <b>{opt_prod_prob:.1f}%</b> 
                    (<span style='color:#2E9E6B; font-weight:700;'>-{diff_prod:.1f}% reduction</span>).
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='insight-card'>Customer holds <b>{sim_num_prod} product(s)</b>, which is within the safe retention range (&le; 2 products).</div>",
                unsafe_allow_html=True,
            )

