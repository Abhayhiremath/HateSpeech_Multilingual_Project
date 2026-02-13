# training/train_lstm.py
import argparse, os, json, time, psutil
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import joblib
import numpy as np

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
class TfIdfDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.values.astype("int64")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32), torch.tensor(self.y[idx])

# ---------------------------------------------------
# Model
# ---------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)

# ---------------------------------------------------
# Train & Eval
# ---------------------------------------------------
def train_loop(model, loader, optim, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    correct = 0; total = 0
    running_loss = 0

    pbar = tqdm(loader, desc="Training batch")
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += len(yb)

        pbar.set_postfix(loss=loss.item(), acc=correct/total)

    return correct/total, running_loss/len(loader)

@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    ys = []; ps = []
    pbar = tqdm(loader, desc="Validating")
    for xb, yb in pbar:
        xb = xb.to(device)
        out = model(xb).cpu()
        ps.extend(out.argmax(1).tolist())
        ys.extend(yb.tolist())
    return ys, ps

# ---------------------------------------------------
# Main
# ---------------------------------------------------
def run(train_csv, val_csv, out_dir, model_name="lstm_model.pt"):
    os.makedirs(out_dir, exist_ok=True)

    print("\n✅ Loading dataset...")
    tr = pd.read_csv(train_csv)
    va = pd.read_csv(val_csv)

    # TF-IDF
    print("\n✅ Building TF-IDF features...")
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=80_000, min_df=2)
    Xtr = vec.fit_transform(tr["text"])
    Xva = vec.transform(va["text"])

    joblib.dump(vec, os.path.join(out_dir, "tfidf_dl.pkl"))

    dtr = TfIdfDataset(Xtr, tr["label"])
    dva = TfIdfDataset(Xva, va["label"])

    tr_loader = DataLoader(dtr, batch_size=256, shuffle=True)
    va_loader = DataLoader(dva, batch_size=512)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMClassifier(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    print("\n✅ Starting LSTM training...\n")
    for epoch in range(1, 7):
        print(f"\n------ Epoch {epoch}/6 ------")
        acc, loss = train_loop(model, tr_loader, opt, device)
        ys, ps = eval_loop(model, va_loader, device)
        print(f"Epoch {epoch}: Train Acc={acc:.3f}, Loss={loss:.3f}")

    # Report
    rep = classification_report(ys, ps, output_dict=True)
    with open(os.path.join(out_dir, "lstm_val_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    torch.save(model.state_dict(), os.path.join(out_dir, model_name))

    print("\n✅ Training complete!")
    print(f"Model saved → {os.path.join(out_dir, model_name)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="models/deep_learning")
    args = ap.parse_args()
    run(args.train, args.val, args.out)
