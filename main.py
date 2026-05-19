from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

# Video Trick: Model ko ek hi baar load karo, memory save hogi
# 'device=-1' ka matlab hai sirf CPU use karo (Server ke liye zaruri)
classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device=-1)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = classifier(image)
    # Sabse zyada confidence wala result bhejo
    return {"label": results[0]['label'], "score": f"{results[0]['score']:.2%}"}