from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Model loading ko simple rakha hai
model = tf.keras.applications.MobileNetV2(weights='imagenet')

@app.get("/")
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Image read
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))
    
    # Preprocessing
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction (Simple output)
    preds = model.predict(img_array)
    return {"status": "success", "prediction": "Processed"}
