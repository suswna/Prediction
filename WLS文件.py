import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib
import os

model_path = "~/Desktop/2024年未完成/3、博士毕业论文/gbdt_WLS_model.pkl"
model_path = os.path.expanduser(model_path)  # 展开用户目录
gbdt_model = joblib.load(model_path)


# Streamlit UI
st.title("High Weaned Litter Size Prediction")

# Add input fields for the 10 features
l_d21h = st.number_input("l.d21H (%)", min_value=45, value=50)
birth_litter_weight = st.number_input("Birth litter weight (kg)", min_value=0.0, value=16.3)
l_d7bf = st.number_input("l.d7BF (mm)", min_value=10, value=17)
l_d21bf = st.number_input("l.d21BF (mm)", min_value=10, value=17)
l_me = st.number_input("L.ME (MJ/kg)", min_value=0.0, value=12.2)
duration_of_lactation = st.number_input("Duration of lactation (days)", min_value=0, value=21)
l_d1h = st.number_input("l.d1H (%)", min_value=45, value=50)
parity = st.number_input("Parity", min_value=1, value=3)
strain_breed_boars_5 = st.selectbox("Strain and breed of boars_5", options=[1, 0])
Litter_size = st.number_input("litter size", min_value=2, value=10)

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "L.d21H":[l_d21h],
    "Birth litter weight": [birth_litter_weight],
    "L.d7BF":[l_d7bf],
    "L.d21BF":[l_d21bf],
    "L.ME": [l_me],
    "duration of lactation": [duration_of_lactation],
    "L.d1H":[l_d1h],
    "Parity": [parity],
    "Strain and breed of boars_5": [strain_breed_boars_5],
    "Litter size": [Litter_size]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

st.subheader("Predicted High Weaned Litter Size")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20,4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())# LLS-prediction