# Credit Card Fraud Detection

## Overview
This application is a real-time transaction scoring dashboard designed to detect fraudulent credit card transactions. It takes anonymized transaction data (PCA components) along with transaction time and amount, and evaluates the probability of fraud using a machine learning pipeline.

## How it Works
- **Live Scoring Interface:** Users can input transaction details (Time, Amount, and V1-V28 PCA features). The app processes these inputs exactly as the model expects and outputs a confidence score for whether the transaction is genuine or fraudulent.
- **Model Diagnostics:** A dedicated section explains the model's performance, highlighting the Precision-Recall AUC (PR-AUC) metric, which is crucial for highly imbalanced datasets.

## Potential Use Cases
- **Financial Institutions & Banks:** For integrating an algorithmic layer to flag suspicious transactions in real-time before they are approved.
- **E-commerce Platforms:** To minimize chargebacks by identifying potentially fraudulent purchases at checkout.

## Machine Learning Model
- **Algorithm:** XGBoost Classifier integrated with SMOTE (Synthetic Minority Over-sampling Technique) to handle the 99.83% class imbalance.
- **Key Metrics:** Precision-Recall AUC (PR-AUC) of **0.9258**.
- **Preprocessing:** The pipeline expects scaled time and amount features alongside 28 PCA-transformed variables.

## Libraries Used
- `streamlit` (Web framework)
- `pandas` & `numpy` (Data manipulation)
- `joblib` (Model loading)
- `xgboost` (Model algorithm)
- `scikit-learn` (Pipeline and preprocessing)
