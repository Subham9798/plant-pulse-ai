from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Model ko load karo (ensure 'model.h5' file tumhari repo mein hai)
try:
    model = tf.keras.models.load_model('model.h5')
except:
    model = None

@app.get("/")
def home():
    return {"message": "Success! Server is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model file not found!"}
        
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return {"prediction": float(np.max(prediction))}
