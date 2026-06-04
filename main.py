from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Yahan tumhara model logic aayega, abhi ke liye dummy data
    return {"prediction": "Early Blight", "confidence": "95%"}
