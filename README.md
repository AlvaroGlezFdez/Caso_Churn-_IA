# Churn Prediction — Automotive (Car Dealership)

Machine-learning pipeline to predict customer churn at an official car dealership and turn
those predictions into a **profitable retention strategy**. Churn is defined as a customer
going **>400 days without a service visit**. Academic case study (B.Sc. in Mathematical
Engineering, UAX). **Academic-case dataset.**

## Objective

Two phases: (1) predict each customer's churn probability from sales, demographic, vehicle,
maintenance and complaint history; (2) design a retention campaign that stays profitable
under strict business constraints (net margin ≥ 30%).

## Feature engineering — leakage handling

A key part of the work was **removing data-leakage and non-deployable variables** so the
model scores real, future customers honestly. Excluded, with reasons:

- `DAYS_LAST_SERVICE` — defines churn by itself (direct leakage).
- `Revisiones`, `km_ultima_revision`, … — not available for new customers.
- `Margen_eur(_bruto)` — computed post-sale, unavailable at scoring time.
- IDs and high-cardinality fields (`Customer_ID`, `CODIGO_POSTAL`, `TIENDA_DESC`).

Engineered features include `compromiso_score` (loyalty products contracted),
`esfuerzo_financiero` (price / estimated income), and `garantia_expira_pronto` (warranty
about to expire — a critical churn moment).

## Modelling

**Recall is the optimization target** (not accuracy/F1): a missed churner costs the full
customer lifetime value, while an unnecessary campaign costs only a few euros — so
maximizing churner detection is economically optimal.

| Model | Recall | Precision | ROC-AUC | F1 | Verdict |
|---|:---:|:---:|:---:|:---:|---|
| **XGBoost** | **0.96** | 0.26 | 0.90 | 0.42 | **Chosen** |
| Random Forest | 0.91 | 0.28 | 0.89 | 0.43 | Runner-up |
| Logistic Regression | 0.90 | 0.27 | 0.88 | 0.41 | Baseline |
| Gradient Boosting | 0.06 | 0.36 | 0.90 | 0.10 | Discarded (recall) |

Low precision is a deliberate trade-off given the asymmetric cost structure.
**No overfitting**: train/test gap 0.007, 5-fold CV stable (std 0.005).

## Scoring new customers

2024 customers have too little history to have "churned" yet, so they are scored by
**relative risk**: probabilities are normalized by the 95th percentile and bucketed into
action segments (high / medium / low), each mapped to a campaign.

## Commercial strategy & ROI

Campaigns per risk segment (personal call, email + gift, upselling), each satisfying the
**net margin ≥ 30%** constraint, with maintenance cost modelled as `C(n) = BASE·(1+α)^n`.
In the recommended scenario: ~6,900 customers actioned, ~€274k campaign cost, ~€797k net
profit → **ROI ≈ 2.9×**.

## Tech stack

Python · scikit-learn · XGBoost · pandas · Matplotlib · Streamlit

## Repository structure

```
├── notebooks/        # EDA, feature engineering, modelling, new-customer scoring
├── model/            # serialized XGBoost model + params + training columns
├── outputs/          # ROC curves, confusion matrices, feature importance, ROI analysis
├── app.py            # Streamlit dashboard
├── docs/             # assignment brief + dataset metadata
└── data/             # (dataset not shipped — see data/README.md)
```

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Full methodology, feature decisions and the complete results are in the notebooks.

---
*Academic case study · Universidad Alfonso X el Sabio (UAX) · 3rd-year Mathematical Engineering. Academic-case data.*
