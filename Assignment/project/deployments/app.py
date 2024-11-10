import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/random_forest_model.pkl")

st.title("Customer Churn Prediction")

# Create inputs
input_data = {}
input_data['feature1'] = st.number_input('Feature 1')
input_data['feature2'] = st.number_input('Feature 2')

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(pd.DataFrame([input_data]))
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
