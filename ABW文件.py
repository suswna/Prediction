# 编辑：苏家宜
# 开发时间：11:14
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib
import os

model_path = "~/Desktop/2024年未完成/3、博士毕业论文/gbdt_ABW_model.pkl"
model_path = os.path.expanduser(model_path)  # 展开用户目录
gbdt_model = joblib.load(model_path)

# Streamlit UI
st.title("Average Birth Weight Prediction")

# Add input fields for the 10 features
g_adf = st.number_input("G.ADF", min_value=0.0, value=0.05)
g_idf = st.number_input("G.IDF", min_value=0.0, value=0.11)
g_ndf = st.number_input("G.NDF", min_value=0.0, value=0.12)
g_sdf = st.number_input("G.SDF", min_value=0.0, value=0.02)
g_dm = st.number_input("G.DM", min_value=0.0, value=0.89)
g_adfip4 = st.number_input("G.ADFIp4 (kg/d)", min_value=0.0, value=3.06)
g_ash = st.number_input("G.Ash", min_value=0.0, value=0.05)  # Adjust the default value as needed
g_adfip3 = st.number_input("G.ADFIp3 (kg/d)", min_value=0.0, value=2.47)
g_ee = st.number_input("G.EE", min_value=0.0, value=0.05)  # Adjust the default value as needed
g_p = st.number_input("G.P", min_value=0.0, value=0.009)

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "G.ADF":[g_adf],
    "G.IDF":[g_idf],
    "G.NDF":[g_ndf],
    "G.SDF":[g_sdf],
    "G.DM":[g_dm],
    "G.ADFIp4":[g_adfip4],
    "G.Ash":[g_ash],
    "G.ADFIp3":[g_adfip3],
    "G.EE":[g_ee],
    "G.P":[g_p]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

st.subheader("Predicted Average Birth Weight")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20,4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())# ABW-prediction