# 🏦 Loan Default Predictor

> Predict whether a LendingClub borrower will **fully repay** or **charge off** their loan — powered by 5 ML models with a Streamlit web app.

## 🧠 Tech Stack & ML Models
Scikit-learn • ONNX • Streamlit • Decision Tree • Random Forest • AdaBoost • Gradient Boosting • XGBoost

---

## 📌 Problem Statement

LendingClub is the world's largest peer-to-peer lending platform. When a borrower applies for a loan, the platform must assess the **risk of default** — will this person repay or charge off?

This project builds an end-to-end ML pipeline that:
- Analyses 396,000+ historical LendingClub loans
- Engineers features from 27 raw columns
- Trains 5 different ML models and compares their performance
- Deploys the best models in a Streamlit web app for instant risk assessment

---

## 🖥️ Live Demo

**[→ Open App on Streamlit Cloud](https://loan-default-prediction-system-ljqprcljx5mw2pmpgtbl7g.streamlit.app/)**

![App Screenshot](https://via.placeholder.com/900x480/0f1923/38bdf8?text=Loan+Default+Predictor+App)

---

## 📁 Project Structure

```
loan-default-predictor/
│
├── loan-defaulters-prediction.ipynb   ← Full EDA + preprocessing + model training
│
├── app.py                             ← Streamlit web application
├── dt_preprocessor.pkl                ← Fitted preprocessing pipeline
│
├── dt_model.onnx                      ← Decision Tree (ONNX)
├── rf_model.onnx                      ← Random Forest (ONNX)
├── ada_model.onnx                     ← AdaBoost (ONNX)
├── gb_model.onnx                      ← Gradient Boosting (ONNX)
├── xgb_model.json                     ← XGBoost (native format)
│
└── requirements.txt
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Lending Club Dataset](https://www.kaggle.com/datasets/jeandedieunyandwi/lending-club-dataset) |
| Rows | 396,030 loans |
| Features | 27 columns |
| Target | `loan_status` — Fully Paid / Charged Off |
| Class split | ~80% Fully Paid · ~20% Charged Off |

### Key Features Used

| Feature | Description |
|---|---|
| `loan_amnt` | Loan amount applied for ($) |
| `int_rate` | Interest rate on the loan |
| `grade` / `sub_grade` | LendingClub assigned risk grade (A–G) |
| `emp_length` | Borrower employment duration |
| `annual_inc` | Self-reported annual income |
| `dti` | Debt-to-income ratio |
| `revol_util` | Revolving line utilization rate |
| `purpose` | Stated reason for the loan |
| `open_acc` / `total_acc` | Number of open / total credit lines |
| `pub_rec` | Number of derogatory public records |

---

## 🔍 Exploratory Data Analysis

Key findings from the notebook:

- **Interest rate is the strongest predictor** — charged-off loans cluster at higher rates
- **Grade F & G loans default significantly more** than A & B loans
- **Higher DTI correlates with default**, especially above 20
- **Debt consolidation** is the most common loan purpose across both classes
- **Employment length has weak predictive power** — 10+ year employees still default

---

## ⚙️ Preprocessing Pipeline

```
Raw CSV (396k rows, 27 cols)
        ↓
Missing value imputation (mean for numeric, mode for categorical)
        ↓
Feature encoding:
  • term           → integer (36 / 60)
  • grade          → ordinal (A=0 … G=6)
  • sub_grade      → ordinal (A1=0 … G5=34)
  • emp_length     → ordinal (< 1 year=0 … 10+=10)
  • emp_title      → top-20 label encoded, rest → "Other"
  • issue_d        → split into issue_year + issue_month
  • earliest_cr_line → earliest_cr_year
  • pub_rec / mort_acc / pub_rec_bankruptcies → binarised
  • verification_status / home_ownership
    application_type / purpose → label encoded
        ↓
Train / Test split (67% / 33%)
```

---

## 🤖 Models & Results

All models trained on the same preprocessed split. Metrics on the test set (130k rows):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Decision Tree | 80.4% | 0.755 | 0.804 | 0.728 | 0.695 |
| AdaBoost | 80.5% | 0.765 | 0.805 | 0.733 | 0.720 |
| Random Forest | 80.5% | 0.764 | 0.805 | 0.729 | 0.715 |
| Gradient Boosting | 80.7% | 0.766 | 0.807 | 0.748 | 0.729 |
| XGBoost | 80.7% | 0.767 |  0.807 | 0.747 | 0.731 |

### Why ONNX for deployment?

Streamlit Cloud has a ~1 GB RAM limit. Large ensemble models (Random Forest at ~400 MB pkl, Gradient Boosting at ~200 MB pkl) crash the app. Exporting to ONNX reduces file sizes by **5–10×**, making all models deployable.

| Model | pkl size | ONNX / JSON size |
|---|---|---|
| Decision Tree | ~1 MB | ~0.5 MB |
| AdaBoost | ~50 MB | ~8 MB |
| Random Forest | ~400 MB | ~50 MB |
| Gradient Boosting | ~200 MB | ~30 MB |
| XGBoost | ~100 MB | ~15 MB (`.json`) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone <repo-url>
cd Loan-Default-Prediction-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

> The app loads the preprocessor from `dt_preprocessor.pkl` and whichever model you select from the sidebar.

---

## 🗺️ Future Improvements

- [ ] **SHAP explainability** — show which features pushed each prediction toward default or repayment, making the model interpretable for end users
- [ ] **Threshold tuning** — add a slider to adjust the decision boundary (0.5 default) so lenders can tune precision vs recall based on their risk appetite
- [ ] **Batch prediction** — upload a CSV of multiple applicants and download a results file with predictions and probabilities for each row
- [ ] **Hyperparameter tuning** — use Optuna or GridSearchCV to squeeze additional performance out of XGBoost and Gradient Boosting
- [ ] **LightGBM model** — benchmark against XGBoost; LightGBM is often faster to train and competitive in accuracy on tabular data
- [ ] **Class imbalance handling** — experiment with SMOTE oversampling or `class_weight='balanced'` to improve recall on the minority (default) class
- [ ] **Model monitoring** — track prediction drift over time if the app receives real loan applications

---
