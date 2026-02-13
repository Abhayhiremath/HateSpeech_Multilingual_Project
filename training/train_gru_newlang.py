# training/train_gru_newlang_clean.py

import pandas as pd
import torch
import torch.nn as nn
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


def train_gru(lang):

    input_file = f"data/processed/{lang}_clean.csv"
    print(f"\n[INFO] Loading dataset → {input_file}")
    df = pd.read_csv(input_file)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    print("\n📌 TF-IDF vectorizing...")
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_vec = tfidf.fit_transform(X_train).toarray()
    X_test_vec = tfidf.transform(X_test).toarray()

    # Save tfidf
    joblib.dump(tfidf, f"models/{lang}_models/tfidf_gru.pkl")

    # Get REAL input dimension
    input_dim = X_train_vec.shape[1]
    print(f"[INFO] TF-IDF feature dimension = {input_dim}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model with REAL dimension
    model = GRUClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 Training GRU...")
    for epoch in range(20):
        model.train()
        losses = []

        for i in tqdm(range(len(X_train_vec))):
            x = torch.tensor(X_train_vec[i]).float().unsqueeze(0).unsqueeze(0).to(device)
            y = torch.tensor([y_train.iloc[i]]).long().to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch+1} Loss: {sum(losses)/len(losses):.4f}")

    save_path = f"models/{lang}_models/gru_best.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Saved GRU model → {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    args = ap.parse_args()
    train_gru(args.lang)
