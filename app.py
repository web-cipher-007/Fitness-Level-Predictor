import streamlit as st
import joblib
import numpy as np


st.set_page_config(page_title="Fitness Level Prediction", layout="centered")
st.title("🏃‍♂️ Fitness Level Prediction Using KNN")
st.write("Predict a person's fitness/obesity level based on Gender, Age, Height, Weight, and Family History.")


model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")
le_gender = joblib.load("le_gender.joblib")
le_family = joblib.load("le_family.joblib")


st.header("Enter Your Details")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 120, 25)
height = st.slider("Height (meters)", 0.5, 2.5, 1.70)
weight = st.slider("Weight (kg)", 10, 200, 70)
family_history = st.selectbox("Family History of Overweight", ["Yes", "No"])


try:
    gender_encoded = le_gender.transform([gender.lower()])[0]
    family_encoded = le_family.transform([family_history.lower()])[0]
except ValueError:
    st.error("Input value does not match training labels. Please check options.")
    st.stop()


sample = np.array([[gender_encoded, age, height, weight, family_encoded]])
sample_scaled = scaler.transform(sample)


if st.button("Predict Fitness Level"):
    prediction = model.predict(sample_scaled)[0]

    
    if "Normal" in prediction:
        st.success(f"Predicted Fitness Level: {prediction}")
    elif "Obesity" in prediction or "Overweight" in prediction:
        st.warning(f"Predicted Fitness Level: {prediction}")
    else:
        st.info(f"Predicted Fitness Level: {prediction}")

    st.info("Prediction is based on Gender, Age, Height, Weight, and Family History using KNN.")

st.markdown("---")
st.markdown("Hit the gym! Stay healthy!")