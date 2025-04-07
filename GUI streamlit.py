import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Simulated training data (replace this with actual dataset features)
df = pd.DataFrame({
    "area": np.random.randint(500, 5000, 100),
    "proximity_to_city": np.random.uniform(1, 20, 100),
    "infrastructure_score": np.random.uniform(1, 10, 100),
    "population_density": np.random.uniform(100, 1000, 100),
    "crime_rate": np.random.uniform(1, 10, 100),
    "avg_income": np.random.uniform(20000, 100000, 100),
    "school_proximity": np.random.uniform(1, 15, 100)
})

# Train StandardScaler on multiple features
scaler = StandardScaler()
scaler.fit(df)  # Training the scaler correctly

# Save the trained scaler
joblib.dump(scaler, "scaler.pkl")

print("Scaler trained on multiple features and saved successfully!")
import streamlit as st
import numpy as np
import joblib

# Load trained model and encoders
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")  # Ensure this file is updated
    label_encoders = joblib.load("label_encoders.pkl")
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Please ensure the model and scaler files are available.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Land Price Prediction", page_icon="🏡", layout="centered")
st.title("🏡 Land Price Predictor")
st.markdown("### Enter details to predict the future land price")

# User Inputs
location = st.selectbox("Select Location", label_encoders["location"].classes_)
area = st.number_input("Enter Land Area (in sqft)", min_value=100, max_value=100000, step=10)
road_access = st.radio("Road Access", ["Yes", "No"])
proximity_to_city = st.number_input("Proximity to City (km)", min_value=0.1, max_value=50.0, step=0.1)
infrastructure_score = st.slider("Infrastructure Score", 1, 10, 5)
population_density = st.number_input("Population Density (people per sq km)", min_value=50, max_value=10000, step=10)
crime_rate = st.slider("Crime Rate", 1.0, 10.0, 5.0)
avg_income = st.number_input("Average Income (INR)", min_value=10000, max_value=200000, step=500)
school_proximity = st.number_input("Proximity to Schools (km)", min_value=0.1, max_value=30.0, step=0.1)
years = st.number_input("Enter Number of Years to Predict", min_value=1, max_value=50, step=1)

# Data Preprocessing
def preprocess_input(location, area, road_access, proximity_to_city, infrastructure_score, population_density, crime_rate, avg_income, school_proximity):
    """Process user input for model prediction."""
    
    # Encode categorical data
    location_encoded = label_encoders["location"].transform([location])[0]
    road_access_encoded = 1 if road_access == "Yes" else 0

    # Scale numerical features
    numerical_features = np.array([[area, proximity_to_city, infrastructure_score, population_density, crime_rate, avg_income, school_proximity]])
    scaled_features = scaler.transform(numerical_features)[0]

    return np.array([[location_encoded, road_access_encoded, *scaled_features]])

# Prediction Function
def predict_price(location, area, road_access, proximity_to_city, infrastructure_score, population_density, crime_rate, avg_income, school_proximity, years):
    input_features = preprocess_input(location, area, road_access, proximity_to_city, infrastructure_score, population_density, crime_rate, avg_income, school_proximity)
    
    try:
        current_price = model.predict(input_features)[0]
        future_price = current_price * (1 + 0.05) ** years  # Assuming 5% annual price growth
        return future_price
    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        return None

# Prediction Button
if st.button("Predict Price"):
    predicted_price = predict_price(location, area, road_access, proximity_to_city, infrastructure_score, population_density, crime_rate, avg_income, school_proximity, years)
    if predicted_price:
        st.success(f"Predicted Land Price after {years} years: ₹{predicted_price:,.2f}")
