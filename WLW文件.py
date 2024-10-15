# 编辑：苏家宜
# 开发时间：16:03
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib
import os

model_path = "~/Desktop/2024年未完成/3、博士毕业论文/gbdt_WLW_model.pkl"
model_path = os.path.expanduser(model_path)  # 展开用户目录
gbdt_model = joblib.load(model_path)


# Streamlit UI
st.title("Weaned Litter Weight Prediction")

# Add input fields for the 10 features
duration_of_lactation = st.number_input("Duration of lactation (days)", min_value=0, value=21)
parity = st.number_input("Parity", min_value=1, value=3)
birth_litter_weight = st.number_input("Birth litter weight (kg)", min_value=0.0, value=16.5)
l_d21t = st.number_input("L.d21T (℃)", min_value=15, value=26)
l_d7bf = st.number_input("L.d7BF (mm)", min_value=8, value=12)
l_d21bf = st.number_input("L.d21BF (mm)", min_value=8, value=12)
l_d1t = st.number_input("L.d1T (℃)", min_value=16, value=26)
l_d1bf = st.number_input("L.d1BF (mm)", min_value=8, value=16)
gestation_days = st.number_input("Gestation days (days)", min_value=101, value=114)
strain_and_breed_of_boars_6 = st.selectbox("Strain and breed of boars_6", options=[1, 0])

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "duration of lactation": [duration_of_lactation],
    "Parity": [parity],
    "Birth litter weight": [birth_litter_weight],
    "L.d21T": [l_d21t],
    "L.d7BF": [l_d7bf],
    "L.d21BF": [l_d21bf],
    "L.d1T": [l_d1t],
    "L.d1BF": [l_d1bf],
    "Gestation days": [gestation_days],
    "Strain and breed of boars_6": [strain_and_breed_of_boars_6]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

st.subheader("Predicted Weaned Litter Weight")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20,4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())# WLW-prediction