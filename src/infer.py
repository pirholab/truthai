import json, torch, joblib
from pathlib import Path
from .common import clean_text, encode
from .train_lstm import BiLSTMAttn
from .train_transformer import TinyTransformer

import os
# Get the directory of this file and navigate to models
current_dir = Path(__file__).parent
project_root = current_dir.parent
MODELS = project_root / "models" / "checkpoints"
VOCAB = json.load(open(project_root / "models" / "vocab.json"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# choose one of:
MODEL_NAME = "transformer_best.pt"  # or "lstm_best.pt"
MODEL_TYPE = "transformer"          # or "lstm"

# optional baseline for fallback:
BASELINE = joblib.load(project_root / "models" / "baseline_tfidf_lr.joblib")

def load_model():
    if MODEL_TYPE == "lstm":
        m = BiLSTMAttn(vocab_size=len(VOCAB))
    else:
        m = TinyTransformer(vocab_size=len(VOCAB))
    m.load_state_dict(torch.load(MODELS/MODEL_NAME, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

MODEL = load_model()

def predict(text: str) -> dict:
    try:
        x = torch.tensor([encode(clean_text(text), VOCAB, 320)], device=DEVICE)
        with torch.no_grad():
            logit = MODEL(x)
            cls = int(logit.argmax(1).item())
            conf = float(torch.softmax(logit, dim=1).max().item())
        return {"label": "REAL" if cls==1 else "FAKE", "confidence": conf, "source": MODEL_TYPE}
    except Exception:
        # fallback to baseline
        cls = int(BASELINE.predict([text])[0])
        # probability for confidence
        if hasattr(BASELINE, "predict_proba"):
            conf = float(BASELINE.predict_proba([text]).max())
        else:
            conf = 0.5
        return {"label": "REAL" if cls==1 else "FAKE", "confidence": conf, "source": "baseline"}
