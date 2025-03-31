import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained Random Forest model
rf_model = joblib.load("models/heart_disease_model.pkl")

# Load the dataset for reference (optional, only if needed for display)
df = pd.read_csv("heart.csv")

# Streamlit App Title
st.title("Heart Disease Prediction Dashboard")
st.write("This app predicts the likelihood of heart disease based on input parameters.")

# Sidebar Inputs
st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol Level", 100, 400, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.slider("Resting ECG Results (0-2)", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.slider("Slope of ST Segment", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.sidebar.slider("Thalassemia Type (0-3)", 0, 3, 1)
    
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display input parameters
st.subheader("User Input Parameters")
st.write(input_df)

# Make prediction
if st.button("Predict"):  # Only predict when button is clicked
    prediction = rf_model.predict(input_df)
    prediction_proba = rf_model.predict_proba(input_df)
    
    # Display Prediction
    st.subheader("Prediction Result")
    st.write("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected")
    
    # Display Probability
    st.subheader("Prediction Probability")
    st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
