<div align="center">

# Financial Fraud Detection
### Binary Classification on Imbalanced Transaction Data

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

</div>

---

## Overview

Financial fraud remains one of the most critical challenges in the banking sector, with global losses exceeding **$485 billion annually**. This project builds a supervised binary classification pipeline to detect fraudulent transactions in real time using behavioral and transactional signals.

The model is trained on a highly imbalanced dataset and optimized for **recall** — minimizing false negatives is prioritized over raw accuracy, since missing a fraud event carries far greater cost than a false alarm.

---

## Business Problem

> *"A model that scores 99% accuracy on fraud data is often useless — because 99% of transactions are legitimate. The real challenge is catching the 1% that isn't."*

Standard accuracy metrics fail on imbalanced fraud datasets. This project addresses that through targeted class imbalance handling, threshold tuning, and business-aligned evaluation metrics.

---

## Model Performance

> Replace placeholders below with your actual notebook output values.

| Metric | Score |
|---|---|
| Accuracy | `99.2%` |
| Precision | `92%` |
| Recall | `84%` |
| ROC-AUC | `0.97%` |
| False Negative Rate | `28%` |

**Why Recall matters here:** A false negative (missed fraud) costs the bank real money. A false positive (flagged legitimate transaction) only costs a customer service call. Model threshold was tuned accordingly.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [PaySim Synthetic Financial Dataset — Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| Rows | ~6.3 million transactions |
| Fraud Rate | ~0.13% (highly imbalanced) |
| Target | `isFraud` (0 = genuine, 1 = fraud) |

**Key features used:**
- `amount` — transaction value
- `oldbalanceOrg` / `newbalanceOrig` — sender balance before & after
- `oldbalanceDest` / `newbalanceDest` — receiver balance before & after
- `type` — transaction type (TRANSFER, CASH_OUT, etc.)
- Engineered: `balance_delta_orig`, `balance_delta_dest`, `zero_balance_flag`

---
