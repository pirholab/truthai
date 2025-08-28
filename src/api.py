from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .infer import predict
from .multi_infer import predict_comprehensive, get_classification_info

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
    author: str = ""
    timestamp: str = ""
    hasImage: bool = False
    hasVideo: bool = False
    imageCount: int = 0
    contentLength: int = 0

@app.get("/")
def root():
    return {
        "message": "TruthAI Comprehensive Post Classification API", 
        "status": "running", 
        "endpoints": ["/health", "/predict", "/classify", "/classification-info", "/docs"],
        "features": ["fake_news_detection", "post_categorization", "content_type_classification"]
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict_endpoint(p: PredictIn):
    """Legacy endpoint for simple fake/real classification"""
    res = predict(p.text)
    return res

@app.post("/classify")
def classify_endpoint(p: PredictIn):
    """Comprehensive post classification endpoint"""
    print(f"🔍 API: Analyzing post from {p.author}: {p.text[:100]}...")
    res = predict_comprehensive(p.text)
    print(f"🔍 API: Classification result: {res['category']['label']} -> {res['type']['label']}")
    return res

@app.get("/classification-info")
def classification_info():
    """Get information about classification categories and types"""
    return get_classification_info()
