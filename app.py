import streamlit as st
import onnxruntime
import numpy as np

# Load the ONNX model
onnx_model_path = "expert.onnx"

ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Streamlit UI
st.title("ONNX Model Score Prediction")

# Input section
st.write("Enter input data:")
# Add input components here based on your model's input requirements
# For example, you can use st.number_input, st.text_input, etc.

# Example input (replace with your actual input data)
# Modify this part to gather input data from the user
input_data = np.random.randn(1, 4, 84, 84).astype(np.float32)

# Prediction
if st.button("Predict"):
    with st.spinner("Predicting..."):
        score = predict_score(input_data)
    st.success(f"Predicted score: {score}")
