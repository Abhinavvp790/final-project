import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set light background color using custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7fafd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Predictive Maintenance Failure Prediction")
st.write("App loaded successfully!")  # Debug line

try:
    model = joblib.load("model.joblib")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define input fields
st.header("Enter Machine Data")
type_map = {'L': 0, 'M': 1, 'H': 2}
Type = st.selectbox("Type (L/M/H)", options=['L', 'M', 'H'])
Air_temperature = st.number_input("Air Temperature [K]", min_value=250.0, max_value=350.0, value=300.0)
Process_temperature = st.number_input("Process Temperature [K]", min_value=250.0, max_value=350.0, value=310.0)
Rotational_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500)
Torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0)
Tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=100)

# Prepare input for prediction
input_data = np.array([[type_map[Type], Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear]])
st.write("Input data:", input_data)  # Debug line

if st.button("Predict Failure Type"):
    prediction = model.predict(input_data)[0]
    categories = ['No Failure', 'Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Tool Wear Failure', 'Random Failures']
    if prediction < len(categories):
        st.success(f"Predicted Failure Type: {categories[prediction]}")
    else:
        st.error("Prediction out of category range.")

    # Show probability meter
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        st.subheader("Failure Probability Meter")
        for i, cat in enumerate(categories):
            st.progress(probs[i])
            st.write(f"{cat}: {probs[i]*100:.2f}%")
    else:
        st.info("Probability meter not available for this model.")