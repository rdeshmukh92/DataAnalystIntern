#FastAPI backend

from fastapi import FastAPI
import joblib

app = FastAPI()

#Load model
model = joblib.load("E:/MS Ritika/Projects/DataAnalystIntern/Task 09/spam_naive_bayes/models/model.pkl")
vectorizer = joblib.load("E:/MS Ritika/Projects/DataAnalystIntern/Task 09/spam_naive_bayes/models/vectorizer.pkl")

@app.get("/")
def home():
    return {"message" : "Spam Classifier API is running"}

@app.post("/predict")
def predict(text: str):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    return{
        "input_text": text,
        "prediction": prediction
    }


"""
Step 1: Run this code.
Step 2: open terminal and set proper path if not. 
Step 3: run python -m uvicorn app.main:app --reload 
       python -m will open the engine
       uvicorn will start web server
       app.main:app shows folder and file location
Step 4: uvicorn running on http://127.0.0.1:8000
Step 5: open http://127.0.0.1:8000/docs
Step 6: We can check if opur FASTAPI is working fine or not. 
    1. click on Get/ -> try it out -> execute -> shows response as ecpected
    2. click on post/predict -> Try it out -> write the text in request body -> execute-> predict correct text spam or ham
"""