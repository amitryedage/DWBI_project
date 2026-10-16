# Retail Banking Customer Churn Dashboard 🏦

An interactive analytics and retention dashboard for retail banking portfolios, built with **Python**, **Streamlit**, **Pandas**, and **Plotly** (managed with **uv**).

---

## 🎯 Executive Overview

Retail banks face steady customer attrition, but retention teams often lack visibility into **who is leaving**, **why**, and **how much capital leaves with them**. 

This dashboard provides executive and branch-level intelligence:
- **Financial Exposure:** Monitors **Balance at Risk** alongside churn rates.
- **Multidimensional Segmentation:** Deep dives across balance bands, tenure milestones, product counts, and customer activity.
- **Actionable Risk Detection:** Rules-based identification of high-risk customers with one-click CSV export for immediate branch outreach.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`

### 1. Run with `uv` (Recommended)
```bash
# Install dependencies
uv sync

# Launch the Streamlit dashboard
uv run streamlit run app.py
```

### 2. Alternative: Run with standard `pip`
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
streamlit run app.py
```

### 3. Run Exploratory Data Analysis & KPI Audit
```bash
# Run automated EDA (Task 2.4)
python scripts/eda.py

# Run mathematical KPI cross-check audit (Task 7.5)
python scripts/verify_kpis.py
```

### 4. Run Unit Tests
```bash
uv run pytest
# or
pytest
```

---

## 📊 Dashboard Architecture & Pages

| Page | Key Visualizations & Metrics |
|---|---|
| **Page 1: Overview** | KPI metric cards (Churn Rate, Total Customers, Avg Balance, Avg Tenure, Avg Products), Donut chart (Retained vs. Churned), Horizontal bar chart of Churn by City, and Automated Dynamic Insights. |
| **Page 2: Balance Analysis** | Churn rate across 5 balance bands, Overlay histogram of balance distributions (Retained vs. Churned), and Capital Exposure callout card. |
| **Page 3: Tenure Analysis** | Churn rate trajectory by tenure year (0–10 yrs), Stacked bar chart of customer volume across tenure bands. |
| **Page 4: Products Held** | Churn rate by product count (1 to 4 products), Churn rate Heatmap cross-tabulating products held against member activity. |
| **Page 5: Risk & Data** | Actionable High-Risk customer table (sorted by balance), filtered data export, and high-risk action list export to CSV. |
| **Page 6: What-If Simulator & ML** | Predictive churn models (Random Forest / Logistic Regression), feature importance chart, and interactive single-customer retention simulator quantifying risk reduction. |

---

## 🛡️ High-Risk Customer Identification Rule

A customer is flagged for immediate intervention if they meet any of the following criteria:
1. Holds **3 or more products** (signaling dissatisfaction or fragmented engagement).
2. Is an **Inactive Member** with an account balance **> ₹1,00,000** (capital flight risk).
3. Is **Age 50+** with tenure **≤ 2 years** (vulnerable onboarding window).

---

## 🎨 Design System & Theme

Adheres strictly to the PRD light theme guidelines:
- **Base Background:** `#FFFFFF`
- **Secondary / Card Background:** `#F4F6FA`
- **Primary Accent:** `#1F4E9C`
- **Retained Indicator:** `#2E9E6B`
- **Churn / Alert Indicator:** `#D64545`
- **Plotly Theme:** `plotly_white`

---

## 📁 Project Structure

```
DWBI_project/
├── .gitignore               # Ignores CSV datasets, PRD docs, venvs, and cache
├── .streamlit/
│   └── config.toml          # Light theme and server configuration
├── data/                    # Dataset directory
│   └── retail_banking_churn.csv
├── src/
│   ├── __init__.py
│   ├── data.py              # Data loader, validation, and derived bands
│   ├── kpis.py              # KPI math, churn_by aggregations, risk rules
│   └── charts.py            # Plotly light-theme visualization suite
├── tests/
│   ├── __init__.py
│   └── test_kpis.py         # Pytest test suite
├── app.py                   # Main Streamlit application
├── pyproject.toml           # uv project configuration and dependencies
└── README.md
```
