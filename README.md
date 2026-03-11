# Fraud Detection System (Production-Style)

A complete end-to-end fraud detection project using the Kaggle **Credit Card Fraud Detection** dataset.

## Project Structure

```text
fraud-detection/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── app/
│   └── streamlit_app.py
├── reports/
├── requirements.txt
└── README.md
```

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset

1. Download `creditcard.csv` from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place the file at:

```text
data/creditcard.csv
```

## 3) Train & Compare Models

This script performs:
- data cleaning (duplicate removal)
- missing-value handling (median imputation)
- feature scaling (RobustScaler)
- feature engineering (time and amount-derived features)
- imbalance handling (SMOTE + undersampling + class weighting)
- model training/comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SVM
  - Gradient Boosting

```bash
python src/train.py --data-path data/creditcard.csv --model-output models/best_fraud_model.joblib --metrics-output models/model_comparison.csv
```

Outputs:
- `models/best_fraud_model.joblib`
- `models/model_comparison.csv`
- `models/model_comparison_summary.json`

## 4) Evaluate Best Model + Visualizations

Generates:
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Feature Importance

```bash
python src/evaluate.py --data-path data/creditcard.csv --model-path models/best_fraud_model.joblib --output-dir reports
```

## 5) Single Prediction (CLI)

Create a CSV containing one row with the same feature columns as dataset except `Class`, then run:

```bash
python src/predict.py --model-path models/best_fraud_model.joblib --input-csv data/sample_transaction.csv
```

## 6) Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

In the app:
- enter transaction input values
- click **Predict Fraud**
- view fraud probability and label

## Sample Predictions

Example output from `predict.py`:

```json
{"prediction": 0, "label": "Legitimate", "fraud_probability": 0.0174}
```

```json
{"prediction": 1, "label": "Fraud", "fraud_probability": 0.9128}
```

## Notes

- Best model is selected by highest **F1 score** on the held-out test split.
- The pipeline is serialized with `joblib` for production inference.
