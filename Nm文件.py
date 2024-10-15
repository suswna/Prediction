# 编辑：苏家宜
# 开发时间：20:15
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib
import os

model_path = "~/Desktop/2024年未完成/3、博士毕业论文/gbdt_Nm_model.pkl"
model_path = os.path.expanduser(model_path)  # 展开用户目录
gbdt_model = joblib.load(model_path)

# Streamlit UI
st.title("Nitrogen in Milk Prediction")
duration_of_lactation = st.number_input("Duration of lactation (days)", min_value=0, value=21)
birth_litter_weight = st.number_input("Birth litter weight (kg)", min_value=0.0, value=16.3)
parity = st.number_input("Parity", min_value=1, value=3)
l_d7bf = st.number_input("L.d7BF (mm)", min_value=8, value=15)
l_d21t = st.number_input("L.d21T (℃)", min_value=15, value=26)
l_adf = st.number_input("L.ADF", min_value=0.0, value=0.05)
l_d21bf = st.number_input("L.d21BF (mm)", min_value=8, value=15)
l_d1bf = st.number_input("L.d1BF (mm)", min_value=8, value=16)
l_d1t = st.number_input("L.d1T (℃)", min_value=16, value=26)
l_d1h = st.number_input("L.d1H (%)", min_value=44, value=65)

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "duration of lactation": [duration_of_lactation],
    "Birth litter weight": [birth_litter_weight],
    "Parity": [parity],
    "L.d7BF": [l_d7bf],
    "L.d21T": [l_d21t],
    "L.ADF": [l_adf],
    "L.d21BF": [l_d21bf],
    "L.d1BF": [l_d1bf],
    "L.d1T": [l_d1t],
    "L.d1H": [l_d1h]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

st.subheader("Predicted Nitrogen in Milk")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20,4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())# DMm-prediction