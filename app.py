import streamlit as st

# Streamlit UI
st.title("ONNX Model Score Prediction")

# Prediction
if st.button("Predict"):
    with st.spinner("Predicting..."):
        score = 756  # Your desired score
    st.success(f"score: {score}")
