from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Global variable for model
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model('model.h5')
        print("Model successfully loaded!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    return {"class_id": int(class_idx), "confidence": float(np.max(prediction))}
