import torch, torch.nn as nn, pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from .common import TextDataset, build_vocab
from sklearn.metrics import accuracy_score, classification_report
import json

DATA = Path("data/processed")
MODELS = Path("models/checkpoints"); MODELS.mkdir(parents=True, exist_ok=True)

class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hidden=128, num_classes=2, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(hidden*2, 1)
        self.fc = nn.Linear(hidden*2, num_classes)

    def forward(self, x):
        x = self.embed(x)                      # [B, T, E]
        H, _ = self.lstm(x)                    # [B, T, 2H]
        w = torch.softmax(self.attn(H).squeeze(-1), dim=1) # [B, T]
        c = (H * w.unsqueeze(-1)).sum(dim=1)   # [B, 2H]
        return self.fc(c)

def load_split():
    train = pd.read_csv(DATA/"train.csv")
    val = pd.read_csv(DATA/"val.csv")
    test = pd.read_csv(DATA/"test.csv")
    return train, val, test

def train_epoch(model, loader, crit, opt, device):
    model.train(); total=0; correct=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward(); opt.step()
        total += y.size(0); correct += (logits.argmax(1)==y).sum().item()
    return correct/total

@torch.no_grad()
def eval_split(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        ys.extend(y.tolist())
        ps.extend(logits.argmax(1).tolist())
    return accuracy_score(ys, ps), classification_report(ys, ps, digits=4)

def main():
    train, val, test = load_split()
    vocab = build_vocab(train.text.tolist(), vocab_size=40000)
    json.dump(vocab, open("models/vocab.json","w"))
    train_ds = TextDataset(train.text.tolist(), train.label.tolist(), vocab)
    val_ds   = TextDataset(val.text.tolist(),   val.label.tolist(),   vocab)
    test_ds  = TextDataset(test.text.tolist(),  test.label.tolist(),  vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttn(vocab_size=len(vocab)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=64)
    test_dl  = DataLoader(test_ds,  batch_size=64)

    best = 0.0
    

    model.load_state_dict(torch.load(MODELS/"lstm_best.pt", map_location=device))
    test_acc, report = eval_split(model, test_dl, device)
    print("test_acc:", test_acc); print(report)

if __name__ == "__main__":
    main()
