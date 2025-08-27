from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .infer import predict

app = FastAPI(title="TruthAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to your domains in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "TruthAI Fake News Detection API", "status": "running", "endpoints": ["/health", "/predict", "/docs"]}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict_endpoint(p: PredictIn):
    res = predict(p.text)
    return res
