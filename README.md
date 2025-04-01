# Heart Disease Prediction Dashboard

This is a web application built using Streamlit and Plotly to predict heart disease based on a dataset of health attributes. The app allows users to upload a CSV file containing heart disease-related data, perform predictions using a machine learning model, and visualize the results interactively.

## Table of Contents

-   [Heart Disease Prediction Dashboard](#heart-disease-prediction-dashboard)
-   [Table of Contents](#table-of-contents)
-   [About](#about)
-   [Features](#features)
-   [Live Demo](#live-demo)
-   [Installation](#installation)
-   [Requirements](#requirements)

## About

This web application is designed to predict heart disease risk using machine learning. It allows users to upload their own datasets and visualize the results. The application is built using Streamlit for the user interface, Plotly for interactive visualizations, and scikit-learn for the machine learning model.

## Features

-   **Upload Your Own Dataset:** Users can upload a CSV file containing heart disease-related data.
-   **Interactive Visualizations:** Interactive visualizations of the dataset using Plotly.
-   **Heart Disease Prediction:** Heart disease prediction using a Logistic Regression model.
-   **Model Evaluation:** Model evaluation with accuracy score and confusion matrix.
-   **Intuitive User Interface:** Clear and intuitive user interface built with Streamlit.

## Live Demo

You can access the deployed application here: [https://heartdiseasedashboard-a7bm7jhhygggrty5gpdmyx.streamlit.app/] (https://heartdiseasedashboard-a7bm7jhhygggrty5gpdmyx.streamlit.app/)

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/mudit-31/heart-disease-dashboard.git](https://github.com/mudit-31/heart-disease-dashboard.git)
    cd heart-disease-dashboard
    ```

2.  **Set up a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**

    ```bash
    streamlit run app.py
    ```

    This will open the dashboard in your default web browser.

## Requirements

-   Python 3.7+
-   Streamlit
-   Plotly
-   scikit-learn
-   pandas
-   seaborn
-   matplotlib

You can install all required dependencies with the following command:

```bash
pip install -r requirements.txt