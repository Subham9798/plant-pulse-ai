from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Model load (apna path sahi rakhna)
model = tf.keras.models.load_model('model.h5')

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = image.resize((224, 224)) # Model ke hisab se size
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    return {"label": f"Class {class_idx}", "confidence": float(np.max(prediction))}
