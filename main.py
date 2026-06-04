from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Model load karo (Agar model.h5 file hai)
# model = tf.keras.models.load_model('model.h5') 

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Yahan image process karne ka code aayega
    return {"status": "success", "message": "Model load ho raha hai"}
