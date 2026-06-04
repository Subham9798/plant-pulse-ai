from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Frontend connection ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_index():
    return FileResponse("index.html")

# Yahan tumhara prediction wala code rahega (jo pehle tha)
@app.post("/predict")
async def predict():
    return {"status": "Prediction logic here"}
