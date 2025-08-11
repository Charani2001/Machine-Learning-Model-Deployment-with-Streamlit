import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Page Config 
st.set_page_config(
    page_title="Diabetes Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/diabetes.csv")

df = load_data()

# Sidebar Navigation
st.sidebar.title("📚 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Data Exploration", "Visualisations", "Prediction", "Model Performance"]
)

# Styling 
st.markdown("""
<style>
    .title-style {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# 1: DATA EXPLORATION 
if menu == "Data Exploration":
    st.markdown("<div class='title-style'>Dataset Overview</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.write("### Data Types")
    st.dataframe(df.dtypes)

    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("### Filter Data")
    age_range = st.slider("Filter by Age", int(df.Age.min()), int(df.Age.max()), (20, 50))
    glucose_min = st.number_input("Minimum Glucose Level", int(df.Glucose.min()), int(df.Glucose.max()), 80)

    filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) & (df["Glucose"] >= glucose_min)]
    st.dataframe(filtered_df)


# 2: VISUALISATIONS 
elif menu == "Visualisations":
    st.markdown("<div class='title-style'>Data Visualisations</div>", unsafe_allow_html=True)

    chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Correlation Heatmap", "Scatter Plot"])
    
    if chart_type == "Histogram":
        feature = st.selectbox("Select Feature", df.columns[:-1])
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

    elif chart_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        x_feature = st.selectbox("X-axis", df.columns[:-1])
        y_feature = st.selectbox("Y-axis", df.columns[:-1])
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue="Outcome", palette="Set1", ax=ax)
        ax.set_title(f"{y_feature} vs {x_feature}")
        st.pyplot(fig)


# 3: PREDICTION
elif menu == "Prediction":
    st.markdown("<div class='title-style'>Diabetes Prediction App</div>", unsafe_allow_html=True)
    st.markdown("### Enter Patient Details Below:")

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

    st.markdown("### Patient Input Summary")
    st.dataframe(input_df)

    if st.button("Predict Diabetes"):
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error("Positive for Diabetes")
        else:
            st.success("Negative for Diabetes")


# 4: MODEL PERFORMANCE
elif menu == "Model Performance":
    st.markdown("<div class='title-style'>Model Performance</div>", unsafe_allow_html=True)

    try:
        X_test = pd.read_csv("data/X_test.csv")
        y_test = pd.read_csv("data/y_test.csv").squeeze()
    except:
        st.warning("No test dataset found. Please ensure X_test.csv and y_test.csv are in data/ folder.")
        st.stop()

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.2f}")
    col2.metric("F1 Score", f"{f1:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    with st.expander("Classification Report"):
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df)
