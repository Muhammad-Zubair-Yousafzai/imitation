import streamlit as st
import onnxruntime
import numpy as np

# Load the ONNX model
onnx_model_path = r"C:\Users\dania\OneDrive\Desktop\flask\expert.onnx"  # Use raw string literal

ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Define the function to make predictions
def predict_score(input_data):
    # Preprocess input data if needed
    # Perform inference
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    result = ort_session.run([output_name], {input_name: input_data})
    score = result[0]
    return score

# Streamlit UI
st.title("ONNX Model Score Prediction")

# Input section
st.write("Enter input data:")
# Add input components here based on your model's input requirements

# Example input (modify based on your model's input requirements)
input_data = np.random.randn(1, 4, 84, 84).astype(np.float32)

# Prediction
if st.button("Predict"):
    with st.spinner("Predicting..."):
        score = predict_score(input_data)
    st.success(f"Predicted score: {score}")
