# Insurance Premium Calculator

## Overview
This application provides an instant estimate of an individual's annual medical insurance premium based on their personal profile and demographic details. It features a polished, customer-facing interface that offers transparency into insurance pricing.

## How it Works
- **Data Input:** Users input their Age, Gender, BMI, Smoking status, Number of Dependents, and Region.
- **Real-Time Prediction:** The app encodes these inputs on the fly and passes them to a pre-trained Decision Tree model.
- **Results:** It outputs an estimated annual premium along with a confidence range (based on the Mean Absolute Error). It also displays a clean summary of the user's profile.

## Potential Use Cases
- **Insurance Agencies & Brokers:** A lead-generation tool on a company website to give prospective clients immediate, indicative quotes.
- **Personal Finance Planning:** Helping individuals estimate future healthcare costs based on lifestyle changes (e.g., quitting smoking or changing BMI).

## Machine Learning Model
- **Algorithm:** Decision Tree Regressor
- **Parameters:**
  - `min_samples_leaf` = 10
  - `max_depth` = 7
  - `min_samples_split` = 2
  - `criterion` = "absolute_error"
  - `random_state` = 42
- **Training Strategy:** The model dynamically trains and caches itself upon the first run using the provided `insurance.csv` dataset, optimizing for mean absolute error (MAE).

## Libraries Used
- `streamlit` (Web framework)
- `pandas` & `numpy` (Data manipulation)
- `scikit-learn` (Machine learning and model building)
