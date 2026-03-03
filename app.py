import streamlit as st
import pickle
import pandas as pd

# Load pipeline
pipeline = pickle.load(open("models/model_pipeline.pkl", "rb"))

st.title("💳 CreditWise Loan Approval System")
st.write("Enter Applicant Details")

# ========== INPUTS ==========
app_income = st.number_input("Applicant Income", min_value=0.0)
co_income = st.number_input("Coapplicant Income", min_value=0.0)
age = st.number_input("Age", min_value=18)
credit_score = st.number_input("Credit Score", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
dti = st.number_input("DTI Ratio", min_value=0.0)
savings = st.number_input("Savings", min_value=0.0)
collateral = st.number_input("Collateral Value", min_value=0.0)

employment = st.selectbox(
    "Employment Status",
    ["Salaried", "Self-Employed", "Business"]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semi-Urban", "Rural"]
)

education = st.selectbox(
    "Education Level",
    ["Graduate", "Postgraduate", "Undergraduate"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    ["Home", "Car", "Education", "Business"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

employer_category = st.selectbox(
    "Employer Category",
    ["Private", "Government", "Self-Employed"]
)

# ========== PREDICTION ==========
if st.button("Predict Loan Status"):

    input_df = pd.DataFrame({
        "Applicant_Income": [app_income],
        "Coapplicant_Income": [co_income],
        "Age": [age],
        "Credit_Score": [credit_score],
        "Loan_Amount": [loan_amount],
        "DTI_Ratio": [dti],
        "Savings": [savings],
        "Collateral_Value": [collateral],
        "Education_Level": [education],
        "Employment_Status": [employment],
        "Marital_Status": [marital_status],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Gender": [gender],
        "Employer_Category": [employer_category]
    })

    # Predict using pipeline
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("✅ Loan Approved")
        st.write(f"Approval Probability: {round(probability * 100, 2)}%")
    else:
        st.error("❌ Loan Rejected")
        st.write(f"Approval Probability: {round(probability * 100, 2)}%")