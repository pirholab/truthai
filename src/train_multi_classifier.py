import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from .common import build_vocab, encode, clean_text
from sklearn.metrics import accuracy_score, classification_report
import json
import math

DATA = Path("data/processed")
MODELS = Path("models/checkpoints")
MODELS.mkdir(parents=True, exist_ok=True)

# Hierarchical Classification System
class PostClassificationSystem:
    CATEGORIES = {
        0: "News",
        1: "Personal", 
        2: "Entertainment",
        3: "Commercial",
        4: "Educational",
        5: "Opinion"
    }
    
    TYPES = {
        # News types
        0: "Politics", 1: "Sports", 2: "Technology", 3: "Health", 
        4: "Science", 5: "Business", 6: "World News", 7: "Local News",
        # Personal types
        8: "Life Update", 9: "Achievement", 10: "Relationship", 11: "Travel",
        # Entertainment types
        12: "Meme", 13: "Funny Video", 14: "Celebrity", 15: "Music", 
        16: "Movies/TV", 17: "Gaming",
        # Commercial types
        18: "Advertisement", 19: "Product Review", 20: "Service Promotion",
        # Educational types
        21: "Tutorial", 22: "Facts/Tips", 23: "Academic",
        # Opinion types
        24: "Political Opinion", 25: "Social Commentary", 26: "Review/Rating"
    }
    
    AUTHENTICITY = {0: "FAKE", 1: "REAL", 2: "N/A"}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiTaskTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=512, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        
        self.category_head = nn.Linear(d_model, len(PostClassificationSystem.CATEGORIES))
        self.type_head = nn.Linear(d_model, len(PostClassificationSystem.TYPES))
        self.authenticity_head = nn.Linear(d_model, len(PostClassificationSystem.AUTHENTICITY))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.pos(self.embed(x))
        h = self.encoder(x).mean(dim=1)
        h = self.dropout(h)
        
        return {
            'category': self.category_head(h),
            'type': self.type_head(h),
            'authenticity': self.authenticity_head(h)
        }
