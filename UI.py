import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Get current directory
current_dir = os.path.dirname(__file__)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join(current_dir, 'xgboostModel.pkl')
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

# Load schema
def load_schema():
    schema_path = os.path.join(current_dir, 'data/columns_set.json')
    with open(schema_path, 'r') as f:
        cols = json.load(f)
    return cols['data_columns']

# Predict function
def predict(data):
    model = load_model()
    return model.predict(data)[0]

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")

# Input form
with st.form("loan_form"):
    name = st.text_input("Your Name", "")
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    marital_status = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    applicant_income = st.number_input("Applicant Income", min_value=0.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0)
    loan_term = st.number_input("Loan Term", min_value=0.0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("Predict")

if submitted:
    schema_cols = load_schema()

    # Initialize all values to 0
    input_data = {col: 0 for col in schema_cols}

    # Set categorical
    if f"Dependents_{dependents}" in input_data:
        input_data[f"Dependents_{dependents}"] = 1
    if f"Property_Area_{property_area}" in input_data:
        input_data[f"Property_Area_{property_area}"] = 1

    # Numerical & binary
    input_data['ApplicantIncome'] = applicant_income
    input_data['CoapplicantIncome'] = coapplicant_income
    input_data['LoanAmount'] = loan_amount
    input_data['Loan_Amount_Term'] = loan_term
    input_data['Gender_Male'] = 1 if gender == "Male" else 0
    input_data['Married_Yes'] = 1 if marital_status == "Yes" else 0
    input_data['Education_Not Graduate'] = 1 if education == "Not Graduate" else 0
    input_data['Self_Employed_Yes'] = 1 if self_employed == "Yes" else 0
    input_data['Credit_History_1.0'] = credit_history

    # Convert to DataFrame
    df = pd.DataFrame({k: [v] for k, v in input_data.items()}).astype(float)

    # Predict
    result = predict(df)

    if result == 1:
        st.success(f"✅ Dear Mr/Mrs/Ms {name}, your loan is **approved**!")
    else:
        st.error(f"❌ Sorry Mr/Mrs/Ms {name}, your loan is **rejected**.")
