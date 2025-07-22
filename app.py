import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Styling
st.markdown("""
    <style>
        .main { background-color: #f0f8ff; }
        .stButton > button {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
        }
        .prediction-box {
            background-color: #1f77b4;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 12px;
            font-size: 22px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 style='color:#1f77b4;'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.write("Use the sliders to input patient data. The model will predict if the patient is likely to have diabetes.")

# Sidebar input
st.sidebar.header("Input Features")

def user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
    Glucose = st.sidebar.slider('Glucose Level', 0, 200, 120)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin', 0.0, 850.0, 79.0)
    BMI = st.sidebar.slider('BMI', 0.0, 70.0, 25.0)
    DPF = st.sidebar.slider('Diabetes Pedigree Function (DPF)', 0.0, 2.5, 0.5)
    Age = st.sidebar.slider('Age', 18, 90, 33)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DPF,
        'Age': Age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Show input
st.subheader("Input Summary")
st.dataframe(input_df)

# Prediction
prediction = model.predict(input_df)
result = "🟥 Positive for Diabetes" if prediction[0] == 1 else "🟩 Negative for Diabetes"

# Display prediction
st.markdown(f"<div class='prediction-box'>{result}</div>", unsafe_allow_html=True)
