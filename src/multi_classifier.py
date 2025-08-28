import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from .common import TextDataset, build_vocab
import json
import math

# Hierarchical Classification System
class PostClassificationSystem:
    """
    Hierarchical post classification:
    1. Category: News, Personal, Entertainment, Commercial, Educational
    2. Type: Sports, Politics, Technology, Meme, Personal Story, etc.
    3. Authenticity: Real/Fake (only for news content)
    """
    
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
        0: "Politics",
        1: "Sports", 
        2: "Technology",
        3: "Health",
        4: "Science",
        5: "Business",
        6: "World News",
        7: "Local News",
        
        # Personal types
        8: "Life Update",
        9: "Achievement",
        10: "Relationship",
        11: "Travel",
        
        # Entertainment types
        12: "Meme",
        13: "Funny Video",
        14: "Celebrity",
        15: "Music",
        16: "Movies/TV",
        17: "Gaming",
        
        # Commercial types
        18: "Advertisement",
        19: "Product Review",
        20: "Service Promotion",
        
        # Educational types
        21: "Tutorial",
        22: "Facts/Tips",
        23: "Academic",
        
        # Opinion types
        24: "Political Opinion",
        25: "Social Commentary",
        26: "Review/Rating"
    }
    
    AUTHENTICITY = {
        0: "FAKE",
        1: "REAL",
        2: "N/A"  # For non-news content
    }

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
    """Multi-task transformer for hierarchical classification"""
    
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=512, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        
        # Shared transformer encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        
        # Task-specific heads
        self.category_head = nn.Linear(d_model, len(PostClassificationSystem.CATEGORIES))
        self.type_head = nn.Linear(d_model, len(PostClassificationSystem.TYPES))
        self.authenticity_head = nn.Linear(d_model, len(PostClassificationSystem.AUTHENTICITY))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Shared representation
        x = self.pos(self.embed(x))
        h = self.encoder(x).mean(dim=1)  # Global average pooling
        h = self.dropout(h)
        
        # Multi-task outputs
        category_logits = self.category_head(h)
        type_logits = self.type_head(h)
        authenticity_logits = self.authenticity_head(h)
        
        return {
            'category': category_logits,
            'type': type_logits,
            'authenticity': authenticity_logits
        }

class MultiTaskDataset(torch.utils.data.Dataset):
    """Dataset for multi-task learning"""
    
    def __init__(self, texts, categories, types, authenticity, vocab, seq_len=320):
        self.texts = texts
        self.categories = categories
        self.types = types
        self.authenticity = authenticity
        self.vocab = vocab
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        from .common import encode, clean_text
        
        text = clean_text(self.texts[idx])
        encoded = encode(text, self.vocab, self.seq_len)
        
        return {
            'text': torch.tensor(encoded, dtype=torch.long),
            'category': torch.tensor(self.categories[idx], dtype=torch.long),
            'type': torch.tensor(self.types[idx], dtype=torch.long),
            'authenticity': torch.tensor(self.authenticity[idx], dtype=torch.long)
        }

def create_sample_multi_class_data():
    """Create comprehensive sample data for multi-task learning"""
    
    data = []
    
    # News - Politics (Real)
    data.extend([
        {"text": "President announces new economic policy to boost job growth", "category": 0, "type": 0, "authenticity": 1},
        {"text": "Senate passes bipartisan infrastructure bill after months of debate", "category": 0, "type": 0, "authenticity": 1},
        {"text": "Local mayor wins re-election with 65% of the vote", "category": 0, "type": 7, "authenticity": 1},
    ])
    
    # News - Politics (Fake)
    data.extend([
        {"text": "Secret documents reveal president is actually an alien from Mars", "category": 0, "type": 0, "authenticity": 0},
        {"text": "Politicians caught using mind control devices on voters", "category": 0, "type": 0, "authenticity": 0},
    ])
    
    # News - Sports (Real)
    data.extend([
        {"text": "Local team wins championship in overtime thriller", "category": 0, "type": 1, "authenticity": 1},
        {"text": "Olympic athlete breaks world record in swimming competition", "category": 0, "type": 1, "authenticity": 1},
        {"text": "Football season starts next week with exciting matchups", "category": 0, "type": 1, "authenticity": 1},
    ])
    
    # News - Sports (Fake)
    data.extend([
        {"text": "Athlete reveals secret superhuman abilities gained from alien technology", "category": 0, "type": 1, "authenticity": 0},
    ])
    
    # News - Technology (Real)
    data.extend([
        {"text": "New smartphone features advanced AI capabilities", "category": 0, "type": 2, "authenticity": 1},
        {"text": "Scientists develop breakthrough in quantum computing", "category": 0, "type": 2, "authenticity": 1},
    ])
    
    # News - Health (Real)
    data.extend([
        {"text": "New study shows benefits of regular exercise for mental health", "category": 0, "type": 3, "authenticity": 1},
        {"text": "Vaccine trials show promising results against new virus strain", "category": 0, "type": 3, "authenticity": 1},
    ])
    
    # Personal - Life Updates
    data.extend([
        {"text": "Just got promoted at work! So excited for this new opportunity", "category": 1, "type": 8, "authenticity": 2},
        {"text": "Celebrating my 5th wedding anniversary today with my amazing spouse", "category": 1, "type": 10, "authenticity": 2},
        {"text": "Finally graduated from college after 4 years of hard work!", "category": 1, "type": 9, "authenticity": 2},
    ])
    
    # Personal - Travel
    data.extend([
        {"text": "Amazing sunset from my hotel balcony in Hawaii", "category": 1, "type": 11, "authenticity": 2},
        {"text": "Exploring the beautiful streets of Paris on my vacation", "category": 1, "type": 11, "authenticity": 2},
    ])
    
    # Entertainment - Memes
    data.extend([
        {"text": "When you realize it's Monday tomorrow... *crying face*", "category": 2, "type": 12, "authenticity": 2},
        {"text": "Me trying to adult vs me wanting to stay in bed all day", "category": 2, "type": 12, "authenticity": 2},
        {"text": "That moment when you find money in your old jacket pocket", "category": 2, "type": 12, "authenticity": 2},
    ])
    
    # Entertainment - Celebrity
    data.extend([
        {"text": "Celebrity couple announces engagement after 2 years of dating", "category": 2, "type": 14, "authenticity": 2},
        {"text": "Famous actor wins award for outstanding performance in latest movie", "category": 2, "type": 14, "authenticity": 2},
    ])
    
    # Commercial - Advertisements
    data.extend([
        {"text": "Limited time offer: 50% off all products this weekend only!", "category": 3, "type": 18, "authenticity": 2},
        {"text": "Try our new restaurant's signature dish - now available for delivery", "category": 3, "type": 18, "authenticity": 2},
    ])
    
    # Educational - Tips
    data.extend([
        {"text": "5 simple tips to improve your productivity while working from home", "category": 4, "type": 22, "authenticity": 2},
        {"text": "How to properly care for your houseplants during winter months", "category": 4, "type": 21, "authenticity": 2},
    ])
    
    # Opinion - Social Commentary
    data.extend([
        {"text": "I think social media is having a negative impact on young people's mental health", "category": 5, "type": 25, "authenticity": 2},
        {"text": "In my opinion, we need better public transportation in our city", "category": 5, "type": 25, "authenticity": 2},
    ])
    
    return data

if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_multi_class_data()
    
    # Expand data by creating variations
    expanded_data = []
    for i in range(3):  # Create 3x more data
        for item in sample_data:
            expanded_data.append({
                "text": item["text"] + f" Additional context and details for training purposes.",
                "category": item["category"],
                "type": item["type"], 
                "authenticity": item["authenticity"]
            })
    
    # Save to CSV
    df = pd.DataFrame(expanded_data)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/multi_class_data.csv", index=False)
    
    print(f"Created multi-class dataset with {len(df)} samples")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Types: {df['type'].value_counts().to_dict()}")
    print(f"Authenticity: {df['authenticity'].value_counts().to_dict()}")
