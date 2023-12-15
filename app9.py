# 编辑：苏家宜
# 开发时间：09:55
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt  # Import Matplotlib

# Load the GBDT model using joblib
model_path = "/Users/sujiayi/Desktop/gbdt_model_rf纤维哺乳期（原数据）9.pkl"
gbdt_model = joblib.load(model_path)

# Streamlit UI
st.title("Weaned Litter Weight Prediction")

# Add input fields for the 10 features
parity = st.number_input("Parity", min_value=1, value=3)
l_me = st.number_input("L.ME", min_value=0.00, value=12.33)
birth_litter_weight = st.number_input("Birth litter weight", min_value=0.0, value=16.5)
weaned_litter_size = st.number_input("Weaned litter size", min_value=0, value=11)
duration_of_lactation = st.number_input("duration of lactation (days)", min_value=0, value=21)

# Create a DataFrame with the input values
input_data = pd.DataFrame({
    "Parity": [parity],
    "L.ME": [l_me],
    "Birth litter weight": [birth_litter_weight],
    "Weaned litter size": [weaned_litter_size],
    "duration of lactation": [duration_of_lactation]
})

# Make predictions using the GBDT model
prediction = gbdt_model.predict(input_data)

# Display the predicted Birth Litter Weight
st.subheader("Predicted Weaned Litter Weight ")
st.write(prediction[0])

# Explain the prediction using SHAP
explainer = shap.Explainer(gbdt_model)
shap_values = explainer(input_data)
st.subheader("SHAP Force Plot")
plt.figure(figsize=(20, 4))  # Set figure size for better visibility
shap.plots.force(explainer.expected_value[0], shap_values.values[0], input_data, matplotlib=True)
st.pyplot(plt.gcf())
