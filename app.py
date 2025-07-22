import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Diabetes Prediction App")
st.write("Enter patient data to check if they are likely to have diabetes.")

# Sidebar input
st.sidebar.header("Input Features")

def user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin', 0.0, 850.0, 79.0)
    BMI = st.sidebar.slider('BMI', 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DPF', 0.0, 2.5, 0.5)
    Age = st.sidebar.slider('Age', 18, 90, 33)
    
    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Show input
st.subheader("User Input Features")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)
st.subheader("Prediction Result")
st.write("Positive for Diabetes" if prediction[0] == 1 else "Negative for Diabetes")