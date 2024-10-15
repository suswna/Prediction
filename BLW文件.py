import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib
import os

model_path = "~/Desktop/2024年未完成/3、博士毕业论文/gbdt_BLW_model.pkl"
model_path = os.path.expanduser(model_path)  # 展开用户目录
gbdt_model = joblib.load(model_path)


# Streamlit UI
st.title("Birth Litter Weight Prediction")

# Add input fields for the 10 features

g_adfip2 = st.number_input("G.ADFIp2 (kg/d)", min_value=0.0, value=2.61)
g_adfip3 = st.number_input("G.ADFIp3 (kg/d)", min_value=0.0, value=2.47)
g_adfip4 = st.number_input("G.ADFIp4 (kg/d)", min_value=0.0, value=3.06)
g_dm = st.number_input("G.DM", min_value=0.0, value=0.89)
parity = st.number_input("Parity", min_value=1, value=3)
g_adf = st.number_input("G.ADF", min_value=0.0, value=0.06)
g_d110bf = st.number_input("G.d110BF (mm)", min_value=10, value=18)
g_p = st.number_input("G.P", min_value=0.0, value=0.009)
g_d1bf = st.number_input("G.d1BF (mm)", min_value=10, value=16)
g_d60bf = st.number_input("G.d60BF (mm)", min_value=10, value=17)


# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "G.ADFIp2":[g_adfip2],
    "G.ADFIp3":[g_adfip3],
    "G.ADFIp4":[g_adfip4],
    "G.DM":[g_dm],
    "Parity": [parity],
    "G.ADF":[g_adf],
    "G.d110BF":[g_d110bf],
    "G.P":[g_p],
    "G.d1BF":[g_d1bf],
    "G.d60BF":[g_d60bf]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

st.subheader("Predicted Birth Litter Weight")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20,4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())# BLW-prediction