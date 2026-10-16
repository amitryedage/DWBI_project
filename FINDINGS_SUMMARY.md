# Retail Banking Customer Churn Analysis: Executive Findings & Recommendations 📊

**Author:** Yedage Amit Ramesh (4010)  
**Domain:** Retail Banking & Financial Intelligence  
**Dataset:** 10,000 Retail Banking Customers (`retail_banking_churn.csv`)  

---

## 1. Executive Summary

An analysis of 10,000 retail banking accounts revealed a baseline **churn rate of ~20.4%**, representing substantial capital attrition across customer segments. More importantly, churn is not distributed randomly—it concentrates sharply around specific behavioral triggers, product holding configurations, and customer engagement tiers.

| Portfolio Metric | Value |
|---|---|
| **Total Customer Base** | 10,000 accounts |
| **Retained Customers** | ~7,963 (79.6%) |
| **Churned Customers** | ~2,037 (20.4%) |
| **Average Account Balance** | ~₹76,485 |
| **Aggregate Capital Exposure (Balance at Risk)** | **~₹182.5 Crore** |
| **High-Risk Segment Size (Rule-flagged)** | ~2,390 accounts (~23.9% of portfolio) |

---

## 2. Key Churn Drivers & Behavioral Insights

### 🔴 Insight 1: The Multi-Product Paradox (Severe Churn on 3+ Products)
* **Finding:** While customer churn for accounts holding 1 or 2 products remains modest (below 25%), customers holding **3 or 4 products experience catastrophic churn rates exceeding 70% to 80%**.
* **Root Cause:** Aggressive cross-selling without proper customer onboarding or fee transparency creates customer friction. Customers with 3+ products often incur surprise fees, complex maintenance rules, or overlapping product features.

### 🔴 Insight 2: Inactive Wealthy Accounts (Capital Flight)
* **Finding:** Account holders categorized as **inactive members** with balances exceeding **₹1,00,000** churn at more than double the rate of active affluent customers.
* **Impact:** Inactive accounts represent the highest **Balance at Risk**. Dormant high-balance customers quietly transition their primary banking relationships elsewhere before closing the account.

### 🔴 Insight 3: Early-Tenure Friction in Senior Customers
* **Finding:** Customers aged **50+ with tenure ≤ 2 years** exhibit higher-than-average attrition compared to seasoned customers with 5+ years of relationship history.
* **Root Cause:** Sub-optimal digital branch onboarding, complex KYC or mobile banking workflows, and insufficient dedicated relationship management for newly onboarded senior customers.

### 🟡 Insight 4: City-Level Geographic Disparities
* **Finding:** Attrition rates vary meaningfully across regions (e.g., Pune, Mumbai, Nashik, Kolhapur, Solapur, Satara, Sangli). Urban hubs with greater fintech and private bank competition show elevated sensitivity to interest rates and digital app experience.

---

## 3. High-Risk Customer Identification Model

To enable proactive rather than reactive retention, the dashboard implements a deterministic multi-factor risk rule:

$$\text{Risk Flag} = (\text{num\_products} \ge 3) \lor (\text{is\_active} = 0 \land \text{balance} > 100,000) \lor (\text{age} \ge 50 \land \text{tenure} \le 2)$$

* **Target Cohort:** Accounts meeting this criterion have historically accounted for the majority of total capital loss.
* **Operational Action:** Branch managers can filter and download this targeted list in one click via **Page 5 (Risk & Data)** for prioritized outreach.

---

## 4. Strategic Recommendations for Retention Teams

1. **Restructure Cross-Selling Policies:**
   * Stop pushing 3rd and 4th products immediately during account opening.
   * Shift product manager KPIs from "products sold" to "30-day active product engagement".
2. **Launch an Inactive Wealth Re-Engagement Workflow:**
   * Automate alerts to branch relationship managers whenever an account with balance > ₹1,00,000 shows zero transactions over 45 days.
   * Offer concierge relationship banking or personalized wealth management reviews.
3. **Dedicated Onboarding for Senior Depositors:**
   * Provide assisted onboarding calls and simplified mobile banking walkthroughs for new customers aged 50+.
4. **Branch Retention Scorecards:**
   * Equip branch managers in high-churn cities with monthly retention scorecards and localized win-back incentives.
