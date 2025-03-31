import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import altair as alt
from bokeh.plotting import figure

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Shree\Desktop\heart_disease_dashboard\heart.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    try:
        print("Data Head:")
        print(df.head())

        df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
        df['ChestPainType'] = df['ChestPainType'].astype('category').cat.codes
        df['RestingECG'] = df['RestingECG'].astype('category').cat.codes
        df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
        df['ST_Slope'] = df['ST_Slope'].astype('category').cat.codes

        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf)
        recall_rf = recall_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf)

        model_xgb = xgb.XGBClassifier(random_state=42)
        model_xgb.fit(X_train, y_train)
        y_pred_xgb = model_xgb.predict(X_test)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        precision_xgb = precision_score(y_test, y_pred_xgb)
        recall_xgb = recall_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb)

        st.title("Heart Disease Prediction Dashboard")

        st.header("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Random Forest Accuracy", f"{accuracy_rf:.2f}")
        col1.metric("Random Forest Precision", f"{precision_rf:.2f}")
        col1.metric("Random Forest Recall", f"{recall_rf:.2f}")
        col1.metric("Random Forest F1", f"{f1_rf:.2f}")

        col2.metric("XGBoost Accuracy", f"{accuracy_xgb:.2f}")
        col2.metric("XGBoost Precision", f"{precision_xgb:.2f}")
        col2.metric("XGBoost Recall", f"{recall_xgb:.2f}")
        col2.metric("XGBoost F1", f"{f1_xgb:.2f}")

        st.header("Feature Importance (SHAP)")
        explainer_shap = shap.TreeExplainer(model_xgb)
        shap_values = explainer_shap.shap_values(X_test)
        fig_shap = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig_shap)

        st.header("Feature Importance (LIME)")
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['No Heart Disease', 'Heart Disease'], mode='classification')
        sample_idx = st.slider("Select Sample Index for LIME Explanation", 0, len(X_test) - 1, 0)
        exp_lime = explainer_lime.explain_instance(X_test.iloc[sample_idx].values, model_xgb.predict_proba, num_features=10)
        fig_lime = exp_lime.as_pyplot_figure()
        st.pyplot(fig_lime)

        st.header("Data Visualizations (Altair)")
        chart_altair = alt.Chart(df).mark_circle().encode(
            x='Cholesterol',
            y='MaxHR',
            color='HeartDisease',
            tooltip=['Cholesterol', 'MaxHR', 'HeartDisease']
        ).properties(title="Cholesterol vs MaxHR").interactive()
        st.altair_chart(chart_altair, use_container_width=True)

        st.header("Data Visualizations (Bokeh)")
        p = figure(x_axis_label='Cholesterol', y_axis_label='MaxHR', title="Cholesterol vs MaxHR")
        p.circle(df['Cholesterol'], df['MaxHR'], color=df['HeartDisease'].map({0: 'blue', 1: 'red'}))
        st.bokeh_chart(p, use_container_width=True)

        st.header("Age Distribution")
        fig_hist = plt.figure(figsize=(10, 6))
        sns.histplot(df['Age'], kde=True)
        st.pyplot(fig_hist)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")