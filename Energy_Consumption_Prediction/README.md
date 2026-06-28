# ⚡ Energy Consumption Predictor

This is a Streamlit web application that predicts the energy consumption of a building using an advanced XGBoost machine learning model. By adjusting building characteristics and external conditions, you can see how different parameters impact energy usage.

## Features
- **Interactive UI**: A simple, user-friendly interface built with Streamlit.
- **Machine Learning Integration**: Uses a pre-trained XGBoost regression model to estimate energy consumption (in kWh).
- **Customizable Parameters**: 
  - **Building Characteristics**: Building Size, Number of Occupants, Insulation Quality, Number of Appliances, and Inside Temperature.
  - **External Conditions**: Outside Temperature, Humidity, Solar Radiation, Wind Speed, Hour of the Day, and Day of the Week.

## Project Structure
- `app.py`: The main Streamlit application script containing the user interface and prediction logic.
- `xgboost_energy_model.pkl`: The saved XGBoost model pipeline used for predictions.
- `energy_consumption_prediction.csv`: The dataset used for training/analysis.
- `Refined_XGBoost_Energy_Consumption.ipynb`: The Jupyter Notebook containing the data exploration, model training, and refinement process.
- `requirements.txt`: The required Python dependencies for the project.

## How to Run

1. **Activate the virtual environment**:
   Make sure you are in the project directory and activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. **Run the Streamlit application**:
   Execute the following command to start the app locally:
   ```bash
   streamlit run app.py
   ```

3. **Use the App**:
   The app will automatically open in your default web browser (typically at `http://localhost:8501`). Enter your desired parameters and click **Predict Energy Consumption** to get the estimated energy usage.

## Requirements
- Python 3.x
- Streamlit
- Pandas
- Scikit-Learn
- XGBoost
- Joblib

*(See `requirements.txt` for specific versions)*
