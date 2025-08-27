import torch, torch.nn as nn, pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from .common import TextDataset, build_vocab
from sklearn.metrics import accuracy_score, classification_report
import json, math

DATA = Path("data/processed")
MODELS = Path("models/checkpoints"); MODELS.mkdir(parents=True, exist_ok=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_ff=512, num_classes=2, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos(self.embed(x))
        # mean pool over tokens
        h = self.encoder(x).mean(dim=1)
        return self.cls(h)

def run():
    train = pd.read_csv(DATA/"train.csv")
    val   = pd.read_csv(DATA/"val.csv")
    test  = pd.read_csv(DATA/"test.csv")

    vocab = build_vocab(train.text.tolist(), vocab_size=50000)
    json.dump(vocab, open("models/vocab.json","w"))
    train_ds = TextDataset(train.text.tolist(), train.label.tolist(), vocab, seq_len=320)
    val_ds   = TextDataset(val.text.tolist(),   val.label.tolist(),   vocab, seq_len=320)
    test_ds  = TextDataset(test.text.tolist(),  test.label.tolist(),  vocab, seq_len=320)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(vocab_size=len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    train_dl = DataLoader(train_ds, batch_size=48, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64)
    test_dl  = DataLoader(test_ds,  batch_size=64)

    best=0.0
    for ep in range(12):
        model.train()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        # eval
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                ys += y.tolist(); ps += logits.argmax(1).tolist()
        import numpy as np
        acc = (np.array(ys)==np.array(ps)).mean()
        print(f"epoch {ep} val_acc={acc:.4f}")
        if acc>best:
            best=acc
            torch.save(model.state_dict(), MODELS/"transformer_best.pt")

    # test
    model.load_state_dict(torch.load(MODELS/"transformer_best.pt", map_location=device))
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in test_dl:
            x,y = x.to(device), y.to(device)
            logits = model(x); ys += y.tolist(); ps += logits.argmax(1).tolist()
    print("test_acc:", accuracy_score(ys, ps))
    print(classification_report(ys, ps, digits=4))

if __name__ == "__main__":
    run()
