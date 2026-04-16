from fastapi import FastAPI, UploadFile, File 
from PIL import Image
import numpy as np
import joblib

app = FastAPI()
model = joblib.load("model/knn_model.pkl")

def preprocess(image):
    image = image.convert("L") 
    image = np.array(image)

    image = 255-image
    image = (image / 255.0)*16
    image = Image.fromarray(image).resize((8,8))
    
#   image = image.resize((8,8)).convert("L")
    return np.array(image).flatten().reshape(1, -1)

@app.get("/")
def home():
    return {"message": "Digit Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile= File(...)):
    image = Image.open(file.file)
    data = preprocess(image)
    prediction = model.predict(data)
    return{"prediction": int(prediction[0])}