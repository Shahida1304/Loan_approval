import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Load model
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("model.cbm")
    return model

model = load_model()

st.title("Loan Default Prediction ")
st.write("Enter loan applicant details to predict default risk.")

# ------------------------------
# Input form
# ------------------------------
with st.form("loan_form"):
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Income", min_value=1000, max_value=1_000_000, value=50000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=3)
    loan_amnt = st.number_input("Loan Amount", min_value=1000, max_value=1_000_000, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.5)
    loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=100.0, value=20.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, max_value=50, value=5)
    
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "OTHER"])
    cb_person_default_on_file = st.selectbox("Default on File", ["Y", "N"])
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # ------------------------------
    # Preprocess input like training
    # ------------------------------
    input_df = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_emp_length": person_emp_length,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "person_home_ownership": person_home_ownership,
        "cb_person_default_on_file": cb_person_default_on_file,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade
    }])
    
    # One-hot encoding for categorical columns (same as training)
    for col in ["person_home_ownership", "cb_person_default_on_file", "loan_intent", "loan_grade"]:
        input_df[col] = input_df[col].astype(str)
    input_df = pd.get_dummies(input_df)
    
    # Add missing columns with 0 (if not in input)
    for col in model.feature_names_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_]
    
    # Predict
    prob = model.predict_proba(input_df)[:,1][0]
    prediction = int(prob >= 0.5)
    
    # Risk level
    if prob < 0.3:
        risk = "Low Risk"
    elif prob < 0.7:
        risk = "Medium Risk"
    else:
        risk = "High Risk"
    
    st.subheader("Prediction Results")
    st.write(f" Default Prediction: {prediction}")
    st.write(f" Default Probability: {prob:.2f}")
    st.write(f" Risk Level: {risk}")
