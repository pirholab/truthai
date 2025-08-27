import re, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_len=300):
        self.seq_len = seq_len
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ids = [encode(clean_text(t), vocab, seq_len) for t in texts]

    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return torch.tensor(self.ids[i]), self.labels[i]

def build_vocab(corpus, vocab_size=40000, min_freq=2):
    from collections import Counter
    cnt = Counter()
    for t in corpus:
        cnt.update(clean_text(t).split())
    tokens = ["<pad>", "<unk>"] + [w for w,f in cnt.most_common(vocab_size) if f>=min_freq]
    return {w:i for i,w in enumerate(tokens)}

def encode(text, vocab, seq_len):
    idxs = [vocab.get(w, vocab["<unk>"]) for w in text.split()]
    idxs = idxs[:seq_len]
    if len(idxs) < seq_len: idxs += [vocab["<pad>"]] * (seq_len - len(idxs))
    return idxs
