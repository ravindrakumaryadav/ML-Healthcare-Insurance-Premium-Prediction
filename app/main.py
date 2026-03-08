import streamlit as st
import pandas as pd
from prediction_helper import  predict

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

st.title("🛡️ Insurance Premium Predictor")
st.markdown("---")

# ── Personal Info ──────────────────────────────────────────────────────────────
st.subheader("Personal Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    marital_status = st.selectbox("Marital Status", ["Unmarried", "Married"])
    region = st.selectbox("Region", ["Northwest", "Southeast", "Northeast", "Southwest"])

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    number_of_dependants = st.number_input("Number of Dependants", min_value=0, max_value=20, value=0)

st.markdown("---")

# ── Financial Info ─────────────────────────────────────────────────────────────
st.subheader("Financial & Employment")
col3, col4 = st.columns(2)

with col3:
    income_lakhs = st.number_input("Annual Income (Lakhs ₹)", min_value=0.0, value=10.0, step=0.5)
    insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])

with col4:
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Freelancer"])

st.markdown("---")

# ── Health Info ────────────────────────────────────────────────────────────────
st.subheader("Health Profile")
col5, col6 = st.columns(2)

with col5:
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obesity", "Underweight"])
    smoking_status = st.selectbox("Smoking Status", ["No Smoking", "Occasional", "Regular"])
    genetical_risk = st.slider("Genetical Risk Score", min_value=0, max_value=5, value=2)

with col6:
    medical_history = st.selectbox("Medical History", [
        "No Disease",
        "Diabetes",
        "High blood pressure",
        "Thyroid",
        "Heart disease",
        "Diabetes & High blood pressure",
        "Diabetes & Thyroid",
        "Diabetes & Heart disease",
        "High blood pressure & Heart disease",
    ])

st.markdown("---")

# input values from UI
input_dict = {
        "age": age,
        "number_of_dependants": number_of_dependants,
        "income_lakhs": income_lakhs,
        "insurance_plan": insurance_plan,
        "genetical_risk": genetical_risk,
        "gender": gender,
        "region": region,
        "marital_status": marital_status,
        "bmi_category": bmi_category,
        "smoking_status": smoking_status,
        "employment_status": employment_status,
        "medical_history": medical_history,
    }





# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Annual Premium", use_container_width=True):
    prediction= predict(input_dict)
    # ── TODO: Load your model and predict ──
    st.success(f"Estimated Annual Premium: ₹ {prediction:,.0f}")