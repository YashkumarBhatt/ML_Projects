# ML Projects

A collection of machine learning projects — predictive models, classification systems, and interactive Streamlit applications built across different domains.

Each project folder contains the Jupyter notebook, trained model files (`.pkl` / `.joblib`), a Streamlit app where applicable, and a `requirements.txt`.

---

## Projects

### ⚡ [Energy Consumption Prediction](./Energy_Consumption_Prediction)
Predicts building energy consumption (kWh) based on structural characteristics and external conditions. Built with XGBoost regression and deployed as an interactive Streamlit app where users can adjust parameters like insulation quality, occupancy, and weather conditions in real time.

`XGBoost` `Regression` `Streamlit`

---

### 🏥 [Insurance Premium Prediction](./Insurance-Premium-Prediction)
Estimates annual medical insurance premiums based on a user's personal and demographic profile. Uses a Decision Tree regression model that trains and caches on first run. The app outputs a premium estimate along with a confidence range derived from MAE.

`Decision Tree` `Regression` `Streamlit`

---

### 🏦 [Loan Approval Probability](./Loan-Approval-Probability)
A three-tab underwriting and portfolio analysis suite. Evaluates loan applicants with live DTI/LTI indicators and a confidence score, visualizes historical portfolio KPIs, and simulates amortization schedules. Uses a tree-based classification model with a dedicated preprocessing pipeline.

`Classification` `Streamlit` `Plotly`

---

## Stack

`Python` · `Scikit-learn` · `XGBoost` · `Streamlit` · `Pandas` · `NumPy` · `Plotly` · `Joblib`

---

## Structure

Each project folder typically contains:

project-name/

├── app.py                  # Streamlit application

├── notebook.ipynb          # Model development and analysis

├── model.pkl / model.joblib # Trained model

├── README.md               # Project-specific details

└── requirements.txt        # Dependencies

---

*More projects will be added as they are completed.*
