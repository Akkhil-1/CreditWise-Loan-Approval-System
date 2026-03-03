import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load files
model = pickle.load(open("models/model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
ohe = pickle.load(open("models/ohe.pkl", "rb"))

st.title("💳 CreditWise Loan Approval System")
st.write("Enter Applicant Details")

# ========== INPUTS ==========
app_income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")
age = st.number_input("Age")
credit_score = st.number_input("Credit Score")
loan_amount = st.number_input("Loan Amount")
dti = st.number_input("DTI Ratio")
savings = st.number_input("Savings")
collateral = st.number_input("Collateral Value")

employment = st.selectbox("Employment Status",
                          ["Salaried", "Self-Employed", "Business"])

property_area = st.selectbox("Property Area",
                             ["Urban", "Semi-Urban", "Rural"])

education = st.selectbox("Education Level",
                         ["Graduate", "Postgraduate", "Undergraduate"])

# ========== PREDICTION ==========
if st.button("Predict Loan Status"):

    # Create dataframe
    input_df = pd.DataFrame({
        "Applicant_Income": [app_income],
        "Coapplicant_Income": [co_income],
        "Age": [age],
        "Credit_Score": [credit_score],
        "Loan_Amount": [loan_amount],
        "DTI_Ratio": [dti],
        "Savings": [savings],
        "Collateral_Value": [collateral],
        "Employment_Status": [employment],
        "Property_Area": [property_area],
        "Education_Level": [education]
    })

    # Separate categorical columns
    cat_cols = ["Employment_Status", "Property_Area", "Education_Level"]

    encoded = ohe.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())

    # Combine numeric + encoded
    numeric_df = input_df.drop(columns=cat_cols)
    final_df = pd.concat([numeric_df.reset_index(drop=True),
                          encoded_df.reset_index(drop=True)], axis=1)

    # Scale
    scaled = scaler.transform(final_df)

    # Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan Approved")
        st.write(f"Approval Probability: {round(probability*100,2)}%")
    else:
        st.error("❌ Loan Rejected")