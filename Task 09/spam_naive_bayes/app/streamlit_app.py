#Streamlit frontend
import streamlit as st 
import requests

st.title("Spam Email classifier (Navie Bayes)")

text = st.text_area("Enter email content")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            params= {"text":text}
        )

        result = response.json()

        if result["prediction"] == "spam":
            st.error("This is a spam")
        else:
            st.success("This is not a spam")



#open terminal -> 
# run 
# streamlit run "e:\MS Ritika\Projects\DataAnalystIntern\Task 09\spam_naive_bayes\app\streamlit_app.py"