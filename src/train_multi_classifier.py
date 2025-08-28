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

class MultiTaskDataset(torch.utils.data.Dataset):
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
        text = clean_text(self.texts[idx])
        encoded = encode(text, self.vocab, self.seq_len)
        
        return {
            'text': torch.tensor(encoded, dtype=torch.long),
            'category': torch.tensor(self.categories[idx], dtype=torch.long),
            'type': torch.tensor(self.types[idx], dtype=torch.long),
            'authenticity': torch.tensor(self.authenticity[idx], dtype=torch.long)
        }

def create_sample_data():
    data = []
    
    # News - Politics
    data.extend([
        {"text": "President announces new economic policy to boost job growth", "category": 0, "type": 0, "authenticity": 1},
        {"text": "Senate passes bipartisan infrastructure bill after months of debate", "category": 0, "type": 0, "authenticity": 1},
        {"text": "Secret documents reveal president is actually an alien from Mars", "category": 0, "type": 0, "authenticity": 0},
        {"text": "Politicians caught using mind control devices on voters", "category": 0, "type": 0, "authenticity": 0},
    ])
    
    # News - Sports
    data.extend([
        {"text": "Local team wins championship in overtime thriller", "category": 0, "type": 1, "authenticity": 1},
        {"text": "Olympic athlete breaks world record in swimming competition", "category": 0, "type": 1, "authenticity": 1},
        {"text": "Athlete reveals secret superhuman abilities from alien technology", "category": 0, "type": 1, "authenticity": 0},
    ])
    
    # Personal posts
    data.extend([
        {"text": "Just got promoted at work! So excited for this new opportunity", "category": 1, "type": 8, "authenticity": 2},
        {"text": "Celebrating my 5th wedding anniversary today with my amazing spouse", "category": 1, "type": 10, "authenticity": 2},
        {"text": "Amazing sunset from my hotel balcony in Hawaii", "category": 1, "type": 11, "authenticity": 2},
    ])
    
    # Entertainment - Memes
    data.extend([
        {"text": "When you realize it's Monday tomorrow... *crying face*", "category": 2, "type": 12, "authenticity": 2},
        {"text": "Me trying to adult vs me wanting to stay in bed all day", "category": 2, "type": 12, "authenticity": 2},
        {"text": "Celebrity couple announces engagement after 2 years of dating", "category": 2, "type": 14, "authenticity": 2},
    ])
    
    # Commercial
    data.extend([
        {"text": "Limited time offer: 50% off all products this weekend only!", "category": 3, "type": 18, "authenticity": 2},
        {"text": "Try our new restaurant's signature dish - now available for delivery", "category": 3, "type": 18, "authenticity": 2},
    ])
    
    # Educational
    data.extend([
        {"text": "5 simple tips to improve your productivity while working from home", "category": 4, "type": 22, "authenticity": 2},
        {"text": "How to properly care for your houseplants during winter months", "category": 4, "type": 21, "authenticity": 2},
    ])
    
    # Opinion
    data.extend([
        {"text": "I think social media is having a negative impact on young people", "category": 5, "type": 25, "authenticity": 2},
        {"text": "In my opinion, we need better public transportation in our city", "category": 5, "type": 25, "authenticity": 2},
    ])
    
    # Expand data
    expanded_data = []
    for i in range(5):  # 5x expansion
        for item in data:
            expanded_data.append({
                "text": item["text"] + f" Additional context for training variation {i}.",
                "category": item["category"],
                "type": item["type"], 
                "authenticity": item["authenticity"]
            })
    
    return expanded_data

def run():
    # Create sample data
    data = create_sample_data()
    df = pd.DataFrame(data)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Build vocabulary
    vocab = build_vocab(train_df.text.tolist(), vocab_size=50000)
    json.dump(vocab, open("models/multi_vocab.json", "w"))
    
    # Create datasets
    train_ds = MultiTaskDataset(
        train_df.text.tolist(), train_df.category.tolist(), 
        train_df.type.tolist(), train_df.authenticity.tolist(), vocab
    )
    val_ds = MultiTaskDataset(
        val_df.text.tolist(), val_df.category.tolist(),
        val_df.type.tolist(), val_df.authenticity.tolist(), vocab
    )
    test_ds = MultiTaskDataset(
        test_df.text.tolist(), test_df.category.tolist(),
        test_df.type.tolist(), test_df.authenticity.tolist(), vocab
    )
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskTransformer(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Loss functions
    category_criterion = nn.CrossEntropyLoss()
    type_criterion = nn.CrossEntropyLoss()
    authenticity_criterion = nn.CrossEntropyLoss()
    
    # Data loaders
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)
    
    # Training
    best_val_acc = 0.0
    for epoch in range(15):
        model.train()
        total_loss = 0
        
        for batch in train_dl:
            texts = batch['text'].to(device)
            categories = batch['category'].to(device)
            types = batch['type'].to(device)
            authenticity = batch['authenticity'].to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            
            # Multi-task loss
            cat_loss = category_criterion(outputs['category'], categories)
            type_loss = type_criterion(outputs['type'], types)
            auth_loss = authenticity_criterion(outputs['authenticity'], authenticity)
            
            total_loss = cat_loss + type_loss + auth_loss
            total_loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        cat_correct = type_correct = auth_correct = total = 0
        
        with torch.no_grad():
            for batch in val_dl:
                texts = batch['text'].to(device)
                categories = batch['category'].to(device)
                types = batch['type'].to(device)
                authenticity = batch['authenticity'].to(device)
                
                outputs = model(texts)
                
                cat_pred = outputs['category'].argmax(1)
                type_pred = outputs['type'].argmax(1)
                auth_pred = outputs['authenticity'].argmax(1)
                
                cat_correct += (cat_pred == categories).sum().item()
                type_correct += (type_pred == types).sum().item()
                auth_correct += (auth_pred == authenticity).sum().item()
                total += categories.size(0)
        
        cat_acc = cat_correct / total
        type_acc = type_correct / total
        auth_acc = auth_correct / total
        avg_acc = (cat_acc + type_acc + auth_acc) / 3
        
        print(f"Epoch {epoch}: Cat={cat_acc:.3f}, Type={type_acc:.3f}, Auth={auth_acc:.3f}, Avg={avg_acc:.3f}")
        
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            torch.save(model.state_dict(), MODELS/"multi_classifier_best.pt")
    
    # Test evaluation
    model.load_state_dict(torch.load(MODELS/"multi_classifier_best.pt", map_location=device))
    model.eval()
    
    cat_preds, type_preds, auth_preds = [], [], []
    cat_true, type_true, auth_true = [], [], []
    
    with torch.no_grad():
        for batch in test_dl:
            texts = batch['text'].to(device)
            outputs = model(texts)
            
            cat_preds.extend(outputs['category'].argmax(1).cpu().tolist())
            type_preds.extend(outputs['type'].argmax(1).cpu().tolist())
            auth_preds.extend(outputs['authenticity'].argmax(1).cpu().tolist())
            
            cat_true.extend(batch['category'].tolist())
            type_true.extend(batch['type'].tolist())
            auth_true.extend(batch['authenticity'].tolist())
    
    print("\n=== TEST RESULTS ===")
    print(f"Category Accuracy: {accuracy_score(cat_true, cat_preds):.3f}")
    print(f"Type Accuracy: {accuracy_score(type_true, type_preds):.3f}")
    print(f"Authenticity Accuracy: {accuracy_score(auth_true, auth_preds):.3f}")

if __name__ == "__main__":
    run()
