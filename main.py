from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Frontend ko connect karne ke liye zaroori hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API is working perfectly!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Image process karne ka basic code
    image = Image.open(io.BytesIO(await file.read()))
    return {"filename": file.filename, "status": "Image received"}
