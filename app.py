import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set page title & layout
st.set_page_config(page_title="Heart Disease Prediction Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "models", "heart_disease_model.pkl")
rf_model = joblib.load(model_path)

# Load and preprocess data
data_path = os.path.join(os.path.dirname(__file__), "data", "heart.csv")
df = pd.read_csv(data_path)

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Scale numeric features except the target variable
scaler = StandardScaler()
numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
numeric_features.remove("HeartDisease")
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Restore original Age range after scaling
df["Age"] = df["Age"] * scaler.scale_[numeric_features.index("Age")] + scaler.mean_[numeric_features.index("Age")]
df["Age"] = df["Age"].astype(int)  # Ensure Age is an integer

# UI Header
st.markdown("""
    <h1 style='text-align: center; color: white; font-weight: bold;'> ❤️ Heart Disease Prediction Dashboard</h1>
    <h3 style='text-align: center; color: white;'>Visualizing Risk Factors & Trends</h3>
    <h4 style='text-align: center; color: white;'>By Mudit Sharma</h4>
""", unsafe_allow_html=True)

# Define layout sections
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# Correlation Heatmap
fig_heatmap = px.imshow(df.corr(), text_auto=True, color_continuous_scale="blues")
fig_heatmap.update_layout(title="Feature Correlations", width=500, height=500, font=dict(color='white'))
top_left.plotly_chart(fig_heatmap, use_container_width=True)

# Cholesterol Distribution Histogram
fig_chol = px.histogram(df, x="Cholesterol", nbins=30, color_discrete_sequence=['blue'])
fig_chol.update_layout(title="Cholesterol Levels Distribution (mg/dL)", xaxis_title="Cholesterol (mg/dL)", font=dict(color='white'))
top_right.plotly_chart(fig_chol, use_container_width=True)

# Age vs Cholesterol Scatter Plot
fig_scatter = px.scatter(df, x="Age", y="Cholesterol", color="HeartDisease", size_max=10)
fig_scatter.update_layout(title="Age vs Cholesterol", xaxis_title="Age (years)", yaxis_title="Cholesterol (mg/dL)", font=dict(color='white'))
bottom_left.plotly_chart(fig_scatter, use_container_width=True)

# Heart Disease Distribution Pie Chart
fig_pie = px.pie(df, names="HeartDisease", title="Heart Disease Distribution", color_discrete_sequence=['green', 'red'])
fig_pie.update_layout(font=dict(color='white'))
bottom_right.plotly_chart(fig_pie, use_container_width=True)

# Additional Charts for Insights
st.markdown("<h2 style='color: white;'>Deeper Insights</h2>", unsafe_allow_html=True)
demographics_col, risk_factors_col = st.columns(2)

# Gender-wise Heart Disease Cases
fig_bar = px.bar(df, x="Sex_M", y="HeartDisease", title="Heart Disease by Gender", color_discrete_sequence=['purple'])
fig_bar.update_layout(xaxis_title="Male (1 = Yes, 0 = No)", yaxis_title="Heart Disease Cases", font=dict(color='white'))
demographics_col.plotly_chart(fig_bar, use_container_width=True)

# Blood Pressure vs Heart Disease Box Plot
fig_box = px.box(df, x="HeartDisease", y="RestingBP", color="HeartDisease", title="Blood Pressure vs Heart Disease")
fig_box.update_layout(xaxis_title="Heart Disease (1 = Yes, 0 = No)", yaxis_title="Resting Blood Pressure (mmHg)", font=dict(color='white'))
risk_factors_col.plotly_chart(fig_box, use_container_width=True)

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: white;'>Built with ❤️ using Streamlit & Plotly</p>
""", unsafe_allow_html=True)
