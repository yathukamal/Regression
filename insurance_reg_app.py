#import libraries
import streamlit as st
import pandas as pd
import joblib

#Load model once
@st.cache_resource
def load_model():
    return joblib.load("insurance_random_forest_model.joblib")

model = load_model()

st.title("Medical Insurance Cost Predictor")
st.write("Enter your details to estimate annual medical insurance costs.")

#--------- User Inputs -----------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["Female", "Male"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

#-------- Encode inputs that were used in the training ------
region_lower = region.lower()

input_df = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,

    #One-hot encoded sex
    "sex_male": 1 if sex == "Male" else 0,

    #One-hot encoded smoker
    "smoker_yes": 1 if smoker == "Yes" else 0,

    #One-hot encoded region (northeast is the baseline)
    "region_northwest": 1 if region_lower == "northwest" else 0,
    "region_southeast": 1 if region_lower == "southeast" else 0,
    "region_southwest": 1 if region_lower == "southwest" else 0,
}])

#-------- Prediction ---------
if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Annual Medical Cost: £{prediction:,.2f}")