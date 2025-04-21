import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💓 Heart Disease Risk Checker")

st.markdown("Enter your info below and check your heart disease risk:")


age = st.number_input("Your Age", min_value=1, max_value=120, step=1)
sex = st.radio("Sex", options=["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level (mg/dL)", value=200)

fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", options=["Yes", "No"])
fbs = 1 if fbs == "Yes" else 0

restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)

exang = st.radio("Chest Pain During Exercise?", options=["Yes", "No"])
exang = 1 if exang == "Yes" else 0

oldpeak = st.number_input("ST Depression After Exercise", value=1.0)

slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
ca = st.slider("Number of Major Vessels (0–3)", 0, 3, 0)

thal = st.selectbox("Thalassemia Type", [3, 6, 7])


if st.button("Check Risk"):
    user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Based on your input, you may be at risk of heart disease.")
    else:
        st.success("✅ Based on your input, you are unlikely to have heart disease.")
