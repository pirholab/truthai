import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def load():
    fake = pd.read_csv(RAW/"Fake.csv")
    true = pd.read_csv(RAW/"True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    # Combine title + text for richer signal
    df["text"] = (df["title"].fillna("") + " \n " + df["text"].fillna("")).str.strip()
    return df[["text","label"]].sample(frac=1.0, random_state=42).reset_index(drop=True)


