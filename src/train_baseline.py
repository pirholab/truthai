import joblib, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA = Path("data/processed")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

def load_split():
    return (
        pd.read_csv(DATA/"train.csv"),
        pd.read_csv(DATA/"val.csv"),
        pd.read_csv(DATA/"test.csv"),
    )

def main():
    train, val, test = load_split()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=80000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
    ])
    pipe.fit(train.text, train.label)
    for name, split in [("val", val), ("test", test)]:
        preds = pipe.predict(split.text)
        print(f"{name} accuracy:", accuracy_score(split.label, preds))
        print(classification_report(split.label, preds, digits=4))
    joblib.dump(pipe, MODELS/"baseline_tfidf_lr.joblib")
    print("Saved to models/baseline_tfidf_lr.joblib")

if __name__ == "__main__":
    main()
