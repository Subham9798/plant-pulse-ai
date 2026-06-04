from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Yahan main change hai: "/" par ab JSON nahi, HTML file dikhegi
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# AI Prediction route
@app.post("/predict")
async def predict():
    return {"status": "Prediction logic will go here"}
