#!/usr/bin/env python3
"""
Quick training script to create baseline models for the TruthAI extension.
This creates minimal working models for demonstration purposes.
"""

import json
import joblib
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from common import clean_text, build_vocab, TextDataset, encode
from train_lstm import BiLSTMAttn
from train_transformer import TinyTransformer

def create_sample_data():
    """Create sample training data if raw data doesn't exist"""
    fake_samples = [
        "BREAKING: Scientists discover that vaccines contain microchips for mind control!",
        "SHOCKING: Government hiding alien technology in secret underground bases!",
        "URGENT: New study proves that 5G towers cause cancer and brain damage!",
        "EXCLUSIVE: Celebrity caught in massive scandal that media won't report!",
        "WARNING: This common food item is actually poisoning your family!",
        "LEAKED: Secret documents reveal government conspiracy to control population!",
        "MIRACLE CURE: This one weird trick doctors don't want you to know!",
        "EXPOSED: Big pharma hiding natural cures to keep you sick and dependent!",
        "ALERT: New world order planning to take over using fake pandemic!",
        "REVEALED: Ancient aliens built pyramids using advanced technology!"
    ]
    
    real_samples = [
        "The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting.",
        "Scientists at MIT published new research on renewable energy storage solutions.",
        "The World Health Organization updated guidelines for COVID-19 prevention measures.",
        "Local school district announces new STEM education program for elementary students.",
        "Weather service issues severe thunderstorm warning for the metropolitan area.",
        "City council approves budget for new public transportation infrastructure project.",
        "University researchers develop new method for early cancer detection screening.",
        "Environmental agency reports improvement in air quality following new regulations.",
        "Technology company announces partnership with nonprofit for digital literacy program.",
        "Agricultural department releases seasonal crop yield forecasts for farmers."
    ]
    
    # Create fake and real dataframes
    fake_df = pd.DataFrame({
        'text': fake_samples,
        'label': [0] * len(fake_samples)  # 0 for fake
    })
    
    real_df = pd.DataFrame({
        'text': real_samples, 
        'label': [1] * len(real_samples)  # 1 for real
    })
    
    return pd.concat([fake_df, real_df], ignore_index=True)

def train_baseline_model(df):
    """Train a simple TF-IDF + Logistic Regression baseline"""
    print("Training baseline TF-IDF model...")
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    # Train logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Create pipeline for easy prediction
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', model)
    ])
    
    # Test accuracy
    predictions = pipeline.predict(df['clean_text'])
    accuracy = accuracy_score(y, predictions)
    print(f"Baseline model accuracy: {accuracy:.3f}")
    
    return pipeline

def create_dummy_neural_models(vocab):
    """Create minimal neural network models with random weights"""
    print("Creating dummy neural network models...")
    
    vocab_size = len(vocab)
    
    # Create LSTM model
    lstm_model = BiLSTMAttn(vocab_size=vocab_size)
    lstm_model.eval()
    
    # Create Transformer model  
    transformer_model = TinyTransformer(vocab_size=vocab_size)
    transformer_model.eval()
    
    return lstm_model, transformer_model

def main():
    """Main training function"""
    print("Setting up TruthAI models...")
    
    # Create directories
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    
    # Load or create data
    data_path = Path("data/raw")
    try:
        if (data_path / "Fake.csv").exists() and (data_path / "True.csv").exists():
            print("Checking existing data files...")
            fake_df = pd.read_csv(data_path / "Fake.csv")
            real_df = pd.read_csv(data_path / "True.csv")
            
            # Check if files have content
            if len(fake_df) == 0 or len(real_df) == 0:
                raise ValueError("Empty CSV files")
                
            # Assume the CSV has a 'text' column, adjust as needed
            if 'text' not in fake_df.columns:
                # Try common column names
                text_col = None
                for col in ['title', 'content', 'article', 'news']:
                    if col in fake_df.columns:
                        text_col = col
                        break
                if text_col:
                    fake_df['text'] = fake_df[text_col]
                    real_df['text'] = real_df[text_col]
                else:
                    raise ValueError("Could not find text column")
            
            fake_df['label'] = 0
            real_df['label'] = 1
            df = pd.concat([fake_df, real_df], ignore_index=True)
            print(f"Loaded {len(df)} samples from existing data")
    except (pd.errors.EmptyDataError, ValueError, FileNotFoundError):
        print("Creating sample data...")
        df = create_sample_data()
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(df['text'].tolist(), vocab_size=1000)
    
    # Save vocabulary
    with open("models/vocab.json", "w") as f:
        json.dump(vocab, f)
    print(f"Saved vocabulary with {len(vocab)} tokens")
    
    # Train baseline model
    baseline = train_baseline_model(df)
    joblib.dump(baseline, "models/baseline_tfidf_lr.joblib")
    print("Saved baseline model")
    
    # Create dummy neural models (for demo purposes)
    lstm_model, transformer_model = create_dummy_neural_models(vocab)
    
    # Save model weights
    torch.save(lstm_model.state_dict(), "models/checkpoints/lstm_best.pt")
    torch.save(transformer_model.state_dict(), "models/checkpoints/transformer_best.pt")
    print("Saved neural network models")
    
    print("\n✅ Setup complete! You can now run the API server.")
    print("Next steps:")
    print("1. Start the API: uvicorn src.api:app --reload")
    print("2. Load the Chrome extension from the webext/ folder")
    print("3. Visit Facebook to test the extension")

if __name__ == "__main__":
    main()
