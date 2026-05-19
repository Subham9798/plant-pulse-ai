from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import torch.nn.functional as F
import io
import os

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load AI
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Correct Pathing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            score, idx = torch.max(probs, dim=-1)
            label = model.config.id2label[idx.item()]
        return {"disease": label, "confidence": f"{score.item()*100:.2f}%", "remedy": "Consult an expert."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))