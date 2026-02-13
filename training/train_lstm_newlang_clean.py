# training/train_lstm_newlang_clean.py
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import joblib


# -----------------------------
# LSTM Model
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, 1, input_dim)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# -----------------------------
# Training Function
# -----------------------------
def train_lstm(lang):
    data_file = f"data/processed/{lang}_clean.csv"

    print(f"\n📂 Loading dataset → {data_file}")
    df = pd.read_csv(data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42
    )

    # ---------------- TF-IDF ----------------
    print("\n🔧 TF-IDF Vectorizing...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train).toarray()
    X_test_vec = tfidf.transform(X_test).toarray()

    input_dim = X_train_vec.shape[1]
    print(f"[INFO] TF-IDF feature dimension = {input_dim}")

    # Save TF-IDF model
    model_dir = f"models/{lang}_models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(tfidf, f"{model_dir}/tfidf_lstm.pkl")

    # ---------------- DataLoader ----------------
    train_data = TensorDataset(torch.tensor(X_train_vec).float(),
                               torch.tensor(y_train.values).long())

    test_data = TensorDataset(torch.tensor(X_test_vec).float(),
                              torch.tensor(y_test.values).long())

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    # ---------------- Model Setup ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training on: {device.upper()}")

    model = LSTMClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    patience = 3
    wait = 0

    print("\n📘 Starting LSTM training...\n")

    # ---------------- Training Loop ----------------
    for epoch in range(20):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(1).to(device)  # (batch, 1, input_dim)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            torch.save(model.state_dict(), f"{model_dir}/lstm_best.pt")
        else:
            wait += 1
            if wait >= patience:
                print("\n⛔ Early stopping triggered!")
                break

    print(f"\n💾 Best model saved at: {model_dir}/lstm_best.pt")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True,
                        help="Language code (e.g. arabic, spanish)")
    args = parser.parse_args()

    train_lstm(args.lang)
