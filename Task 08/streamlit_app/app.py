import streamlit as st 
import requests

st.title("Digit Classifier")

file = st.file_uploader("Upload Image", type = ["png", "jpg"])
if file:
    st.image(file)

    if st.button("predict"):
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files= {"file": file.getvalue()}
        )
        st.write(response.json())

        st.write("Prediction:", response.json().get("Prediction"))