from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import uvicorn

app = FastAPI(title="TruthAI API - Simple Mock")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    # Simple mock prediction based on text length and keywords
    text = p.text.lower()
    
    # Mock logic: longer posts more likely to be real, certain keywords trigger fake
    fake_keywords = ["breaking", "urgent", "shocking", "unbelievable", "doctors hate", "secret"]
    real_keywords = ["today", "according to", "research shows", "study finds"]
    
    fake_score = sum(1 for keyword in fake_keywords if keyword in text)
    real_score = sum(1 for keyword in real_keywords if keyword in text)
    
    # Add some randomness
    base_score = random.uniform(0.6, 0.9)
    
    if fake_score > real_score:
        label = "FAKE"
        confidence = min(0.95, base_score + (fake_score * 0.1))
    else:
        label = "REAL"
        confidence = min(0.95, base_score + (real_score * 0.05))
    
    return {
        "label": label,
        "confidence": confidence,
        "source": "mock_api"
    }

@app.post("/feedback")
def feedback_endpoint(feedback_data: dict):
    return {"status": "received", "message": "Thank you for your feedback!"}

if __name__ == "__main__":
    print("🚀 Starting TruthAI Mock API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
