import streamlit as st
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection App")

st.write("Enter transaction features (30 values):")

features = []
for i in range(30):
    val = st.number_input(f"Feature {i+1}", step=0.01, format="%.4f")
    features.append(val)

if st.button("Check for Fraud"):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Transaction is FRAUDULENT (Confidence: {probability:.2%})")
    else:
        st.success(f"Transaction is NOT fraudulent (Confidence: {1 - probability:.2%})")

