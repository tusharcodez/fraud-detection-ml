<div align="center">

# Financial Fraud Detection
### Ensemble ML Models on Imbalanced Transaction Data

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-0.9973_AUC-brightgreen?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

</div>

---

## Overview

Financial fraud causes billions in annual losses globally. This project builds a supervised binary classification pipeline to detect fraudulent transactions using three ensemble models — XGBoost, Random Forest, and LightGBM — trained on a highly imbalanced dataset with full EDA, feature engineering, and business-aligned evaluation.

---

## Business Problem

> *"A model that scores 99% accuracy on fraud data is often useless — because 99% of transactions are legitimate. The real challenge is catching the 1% that isn't."*

Standard accuracy metrics fail on imbalanced fraud datasets. This project addresses that through targeted class imbalance handling, threshold tuning, and recall-prioritized evaluation.

---

## Model Performance

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | 0.96 | 0.95 | **0.96** | **0.9973** |
| Random Forest | 0.97 | 0.94 | 0.95 | 0.9690 |
| LightGBM | 0.96 | 0.95 | 0.95 | 0.9974 |

> XGBoost selected as the final model — best balance of recall and F1 on fraud class.

**Why Recall matters here:** A false negative (missed fraud) costs real money. A false positive only costs a customer service call. Models were evaluated with that asymmetry in mind.

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

## Methodology
## Methodology

```text
Raw Data
   └── EDA & Class Imbalance Analysis (0.13% fraud rate)
         └── Feature Engineering (balance deltas, zero-balance flags)
               └── Class Imbalance Handling
                     └── Model Training — XGBoost / Random Forest / LightGBM
                           └── Threshold Optimization (Precision-Recall curve)
                                 └── Evaluation (Confusion Matrix, ROC-AUC, F1)
                                       └── Model Comparison & Final Selection
```

**Why ensemble models?**
- Capture complex non-linear relationships between features
- Robust to outliers and skewed distributions
- Native support for class imbalance via `scale_pos_weight` (XGBoost) and `class_weight` (RF)
- Probability outputs enable flexible threshold tuning per business tolerance
