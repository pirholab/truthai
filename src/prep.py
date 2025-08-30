import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)



def main():
    df = load()
    train_df, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)
    train_df.to_csv(OUT/"train.csv", index=False)
    val_df.to_csv(OUT/"val.csv", index=False)
    test_df.to_csv(OUT/"test.csv", index=False)
    print(train_df.shape, val_df.shape, test_df.shape)

if __name__ == "__main__":
    main()
