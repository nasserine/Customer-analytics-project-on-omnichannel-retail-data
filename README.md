# Omnichannel Customer Analytics — RetailHive Dataset

> End-to-end customer analytics project : exploratory analysis, behavioral segmentation and propensity scoring on an omnichannel retail dataset (2019–2025).

---

## Context

RetailHive is an omnichannel retailer operating across 3 sales channels — **Mobile**, **Online** and **Store** — in 3 regions (US, Europe, Asia).

**Business problem identified** : revenue has been flat at ~180k€/year since 2019, and 85.7% of customers buy only once and never return. The company needs to understand its customer base and prioritize retention efforts.

**3 analytical objectives :**

| # | Objective | Deliverable |
|---|-----------|-------------|
| 1 | Diagnose the current state of the customer base | EDA + visualizations |
| 2 | Segment customers by behavioral profile | 3 actionable segments |
| 3 | Score one-shot customers by repurchase propensity | Prioritized reactivation list |

---

## Dataset

**Source** : [RetailHive Pro — Kaggle](https://www.kaggle.com/datasets/ehsanzx/omni-channel-sales-dataset-20192025)

| Field | Value |
|-------|-------|
| Transactions | 12,000 |
| Unique customers | 10,358 |
| Period | 2019-01-01 → 2025-12-31 |
| Channels | Mobile, Online, Store |
| Regions | US, Europe, Asia |
| Products | 100 |
| Total revenue | 1,240,598 € |

**Columns :**

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique transaction identifier |
| `customer_id` | string | Unique customer identifier |
| `product_title` | string | Product name |
| `date` | date | Transaction date |
| `channel` | string | Sales channel (Mobile / Online / Store) |
| `region` | string | Customer region (US / Europe / Asia) |
| `quantity` | int | Number of items purchased |
| `price` | float | Unit price |
| `revenue` | float | Total revenue for the transaction |
| `rating` | string | Product rating (One to Five) |

---

## Project Structure

```
omnichannel-analytics-retailhive/
│
├── omnichannel_analysis_retailhive.ipynb   # Main notebook — full analysis
├── retailhive_10k.csv                      # Raw dataset (from Kaggle)
├── README.md                               # This file
│
├── outputs/
│   ├── master_customer_segmented.csv       # Enriched customer table (1 row per customer)
│   └── liste_reactivation_oneshot.csv      # Scored one-shot customers
│
└── viz/
    ├── viz_01_ca_annuel.png                # Revenue evolution 2019–2025
    ├── viz_02_frequence.png                # Purchase frequency distribution
    ├── viz_03_rating.png                   # Satisfaction by channel and region
    ├── viz_04_features.png                 # Feature engineering overview
    ├── viz_05_segments.png                 # 3-segment visualization
    └── viz_06_modele.png                   # ROC curve + feature importance
```

---

## Methodology

### 1. Exploratory Data Analysis

- Revenue trend analysis by channel and year
- Purchase frequency distribution
- Customer satisfaction analysis (rating) by channel and region

**Key findings :**
- Revenue flat at ~180k€/year — no organic growth
- **85.7% of customers buy only once** — retention is the #1 issue
- Average rating of 2.9/5, uniform across all channels and regions — systemic problem

---

### 2. Feature Engineering

Construction of a **master_customer table** (1 row per customer, 16 variables) from 4 independent variable groups :

| Group | Variables |
|-------|-----------|
| **RFM** | recency, frequency, monetary, monetary_mean, customer_age_days |
| **Omnichannel behavior** | nb_canaux, canal_prefere, pct_mobile, pct_online, pct_store |
| **Satisfaction** | rating_moyen, rating_last, rating_evolution |
| **Profile** | region |

---

### 3. Customer Segmentation

**Why not K-Means ?**

K-Means was tested (k=2 to k=6). The best silhouette score was obtained at k=4 (0.372), but the 4 clusters produced were not actionable : 3 of the 4 clusters contained quasi-identical one-shot customers differentiated only by their sales channel, not by behavior or value. K-Means cannot create meaningful groups when 86% of customers share the same profile (1 purchase, 1 channel).

**Approach chosen : explicit rule-based segmentation**

Two independent axes were defined :

- **Loyalty axis** : `frequency >= 2` → did the customer return ?
- **Omnichannel axis** : `nb_canaux >= 2` → did the customer use multiple channels ?

> **Important structural note** : in this dataset, having 2+ channels implies 2+ purchases by definition. A customer with only 1 transaction can only use 1 channel. The "omnichannel one-shot" profile does not exist in the data.

**3 resulting segments :**

| Segment | Customers | % Base | Avg Revenue | % Total Revenue |
|---------|-----------|--------|-------------|-----------------|
| **Ambassadeur** (loyal + omnichannel) | 1,009 | 9.7% | 218€ | 17.7% |
| **Fidèle Mono-canal** (loyal, 1 channel) | 476 | 4.6% | 214€ | 8.2% |
| **One-shot** (1 purchase, 1 channel) | 8,873 | 85.7% | 104€ | 74.0% |

**Key insight** : Fidèle Mono-canal (214€) and Ambassadeur (218€) have near-identical average revenue.
**Loyalty drives value, not omnichannel behavior alone.**

---

### 4. ML Model — Repurchase Propensity Scoring

**Objective** : among the 8,873 one-shot customers, identify those most likely to make a second purchase.

**Anti-data leakage rule** : features are built exclusively from the **first purchase only**. Using post-purchase data (frequency, nb_canaux, etc.) to predict whether a customer returned would be data leakage and produce a falsely perfect AUC of 1.0.

| Feature | Description |
|---------|-------------|
| `revenue_first` | Revenue of the first purchase |
| `rating_first` | Rating given after the first purchase |
| `canal_enc` | Channel used for the first purchase |
| `region_enc` | Customer region |

**Model** : Gradient Boosting Classifier — robust on imbalanced datasets.

**Results :**

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.497 |
| AUC cross-val (5-fold) | 0.521 ± 0.012 |

**Interpretation** : AUC ~ 0.50 means the model performs no better than random.
This is not an algorithm failure — it is a data limitation.
The first purchase alone (revenue, rating, channel, region) does not explain why a customer returns.
Return behavior depends on factors absent from the dataset : post-purchase email quality, delivery experience, customer service, browsing behavior.

**Recommendation to the client** : enrich data collection (email opens, site navigation, customer service interactions) to enable a genuinely predictive model.

**What the scoring still provides** : even with low AUC, the predicted probabilities allow **prioritizing** the reactivation list. Contacting the top-priority segment rather than all 8,873 one-shot customers at random improves marketing ROI.

| Priority | Customers | Assumed conversion | Recoverable revenue |
|----------|-----------|-------------------|---------------------|
| High priority | ~2,957 | 5% | ~33,000€ |
| Medium priority | ~2,958 | 2% | ~13,000€ |
| Low priority | ~2,958 | 0.5% | ~3,000€ |
| **Total** | **8,873** | | **~49,000€** |

---

## Key Recommendations

**1. Protect loyal customers (priority : HIGH)**
Ambassadeurs and Fidèles Mono-canal represent 14.3% of the base but 25.9% of total revenue. Invest in loyalty programs and personalized offers before they churn.

**2. Trigger the second purchase (priority : HIGH)**
Send a targeted email 14 days after the first purchase with a complementary product recommendation. Even 5% conversion on the high-priority segment = significant revenue impact.

**3. Investigate the low rating (priority : MEDIUM)**
A 2.9/5 average uniform across all channels and regions suggests a systemic problem — product quality, delivery, or customer service — not a channel-specific issue.

**4. Enrich behavioral data collection (priority : MEDIUM)**
Email open rates, site browsing behavior, customer service interactions. These are the missing signals needed to build a genuinely predictive repurchase model.

---

## Tech Stack

```
Python 3.10+
pandas          — data manipulation
numpy           — numerical operations
matplotlib      — visualizations
seaborn         — statistical visualizations
scikit-learn    — StandardScaler, KMeans, GradientBoostingClassifier
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/omnichannel-analytics-retailhive.git
cd omnichannel-analytics-retailhive

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Download the dataset from Kaggle and place it in the root folder
# https://www.kaggle.com/datasets/ehsanzx/omni-channel-sales-dataset-20192025

# 4. Launch the notebook
jupyter notebook omnichannel_analysis_retailhive.ipynb
```

---

## Author

**Nasserine Benchettara**  
Data Scientist — Customer Analytics & Omnichannel Retail  
PhD in Artificial Intelligence (Paris 13) | 10+ years experience

[LinkedIn](https://linkedin.com) · [Malt](https://malt.fr)

---

*This project is part of a customer analytics portfolio demonstrating end-to-end data science skills on real retail data.*
