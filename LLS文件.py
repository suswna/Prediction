import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib
import os

model_path = "~/Desktop/2024年未完成/3、博士毕业论文/gbdt_LLS_model.pkl"
model_path = os.path.expanduser(model_path)  # 展开用户目录
gbdt_model = joblib.load(model_path)


# Streamlit UI
st.title("High Live Litter Size Prediction")

# Add input fields for the 10 features
g_adfip2 = st.number_input("G.ADFIp2 (kg/d)", min_value=0.0, value=2.61)
g_adfip3 = st.number_input("G.ADFIp3 (kg/d)", min_value=0.0, value=2.47)
g_adfip4 = st.number_input("G.ADFIp4 (kg/d)", min_value=0.0, value=3.06)
g_d110bf = st.number_input("G.d110BF (mm)", min_value=10, value=18)
parity = st.number_input("Parity", min_value=1, value=3)
g_d30bf = st.number_input("G.d30BF (mm)", min_value=10, value=17)
g_d1bf = st.number_input("G.d1BF (mm)", min_value=10, value=17)
g_d60h = st.number_input("G.d60H (%)", min_value=45, value=50)
g_ee = st.number_input("G.EE", min_value=0.0, value=0.0533)
g_adf = st.number_input("G.ADF", min_value=0.0, value=0.06)

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "G.ADFIp2":[g_adfip2],
    "G.ADFIp3":[g_adfip3],
    "G.ADFIp4":[g_adfip4],
    "G.d110BF":[g_d110bf],
    "Parity": [parity],
    "G.d30BF":[g_d30bf],
    "G.d1BF":[g_d1bf],
    "G.d60H":[g_d60h],
    "G.EE":[g_ee],
    "G.ADF":[g_adf]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

st.subheader("Predicted High Live Litter Size")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20,4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())# LLS-prediction

