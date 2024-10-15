# 编辑：苏家宜
# 开发时间：16:49
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib
import os

model_path = "~/Desktop/2024年未完成/3、博士毕业论文/gbdt_AWW_model.pkl"
model_path = os.path.expanduser(model_path)  # 展开用户目录
gbdt_model = joblib.load(model_path)


# Streamlit UI
st.title("Average Weaning Weight Prediction")

# Input fields for the new features
l_me = st.number_input("L.ME (MJ/kg)", min_value=0.0, value=12.5)
l_p = st.number_input("L.P ", min_value=0.0, value=0.008)
l_ash = st.number_input("L.Ash ", min_value=0.0, value=0.05)
parity = st.number_input("Parity", min_value=1, value=3)
l_cp = st.number_input("L.CP ", min_value=0.0, value=0.14)
rss_3 = st.selectbox("RSS_3", options=[1, 0])
birth_litter_weight = st.number_input("Birth litter weight (kg)", min_value=0.0, value=16.3)
duration_of_lactation = st.number_input("Duration of lactation (days)", min_value=0, value=21)
l_dm = st.number_input("L.DM ", min_value=0.0, value=0.89)
gestation_days = st.number_input("Gestation days (days) ", min_value=101, value=117)

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "L.ME": [l_me],
    "L.P": [l_p],
    "L.Ash": [l_ash],
    "Parity": [parity],
    "L.CP": [l_cp],
    "RSS_3": [rss_3],
    "Birth litter weight": [birth_litter_weight],
    "duration of lactation": [duration_of_lactation],
    "L.DM": [l_dm],
    "Gestation days": [gestation_days]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

st.subheader("Predicted Average Weaning weight")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20,4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())# AWW-prediction