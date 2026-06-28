import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Energy Consumption Predictor")
st.markdown("Predict the energy consumption of your building using our advanced XGBoost machine learning model. Adjust the parameters below to see how they impact energy usage.")

# Load the model pipeline
@st.cache_resource
def load_model():
    return joblib.load("xgboost_energy_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout
st.header("Building Characteristics")
col1, col2 = st.columns(2)

with col1:
    building_size = st.number_input("Building Size (sq units)", min_value=10.0, max_value=5000.0, value=200.0, step=10.0)
    num_occupants = st.number_input("Number of Occupants", min_value=0, max_value=500, value=10, step=1)
    insulation_quality = st.slider("Insulation Quality", min_value=0.0, max_value=3.0, value=1.0, step=0.1)

with col2:
    num_appliances = st.number_input("Number of Appliances", min_value=0, max_value=200, value=15, step=1)
    inside_temp = st.number_input("Inside Temperature (°C)", min_value=5.0, max_value=40.0, value=22.0, step=0.5)

st.header("External Conditions")
col3, col4 = st.columns(2)

with col3:
    outside_temp = st.number_input("Outside Temperature (°C)", min_value=-30.0, max_value=55.0, value=15.0, step=0.5)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    solar_radiation = st.number_input("Solar Radiation (W/m²)", min_value=0.0, max_value=1500.0, value=300.0, step=10.0)

with col4:
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    hour_of_day = st.slider("Hour of Day", min_value=0, max_value=23, value=12, step=1)
    day_of_week = st.slider("Day of Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=2, step=1)
    is_weekend = 1 if day_of_week >= 5 else 0

st.markdown("---")

if st.button("Predict Energy Consumption", type="primary", use_container_width=True):
    # Prepare the input dataframe
    input_data = pd.DataFrame({
        'building_size': [building_size],
        'num_occupants': [num_occupants],
        'num_appliances': [num_appliances],
        'outside_temp': [outside_temp],
        'inside_temp': [inside_temp],
        'humidity': [humidity],
        'hour_of_day': [hour_of_day],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'solar_radiation': [solar_radiation],
        'wind_speed': [wind_speed],
        'insulation_quality': [insulation_quality]
    })
    
    # Predict
    with st.spinner("Predicting..."):
        prediction = model.predict(input_data)[0]
    
    st.success(f"**Predicted Energy Consumption:** {prediction:.2f} kWh")
    st.balloons()
