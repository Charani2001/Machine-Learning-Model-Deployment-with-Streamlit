import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="", layout="centered")

# Load model
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
        .title-style {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title-style'>Diabetes Prediction App</div>", unsafe_allow_html=True)
st.markdown("### Enter Patient Details Below:")

# Input form layout
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17, value=1, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)

with col2:
    Insulin = st.number_input('Insulin', min_value=0.0, max_value=850.0, value=79.0, step=0.1)
    BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    DPF = st.number_input('Diabetes Pedigree Function (DPF)', min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    Age = st.number_input('Age', min_value=18, max_value=90, value=33, step=1)

# Collect input
input_df = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DPF],
    'Age': [Age]
})

# Show input
st.markdown("### Patient Input Summary")
st.dataframe(input_df)

# Predict
if st.button("Predict Diabetes"):
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        result_html = """
        <div style='background-color:#d9534f; color:white; padding:20px; text-align:center;
                    border-radius:12px; font-size:22px; margin-top:20px;'>
            Positive for Diabetes
        </div>
        """
    else:
        result_html = """
        <div style='background-color:#5cb85c; color:white; padding:20px; text-align:center;
                    border-radius:12px; font-size:22px; margin-top:20px;'>
            Negative for Diabetes
        </div>
        """
    
    st.markdown(result_html, unsafe_allow_html=True)
