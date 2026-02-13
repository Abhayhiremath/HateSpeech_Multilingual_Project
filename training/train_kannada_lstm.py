# training/train_kannada_lstm.py
import os, json
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib

TRAIN = r"D:\HateSpeech_Multilingual_Project\data\kannada\train.csv"
VAL   = r"D:\HateSpeech_Multilingual_Project\data\kannada\val.csv"
OUT   = r"D:\HateSpeech_Multilingual_Project\models\kannada"
os.makedirs(OUT, exist_ok=True)

class TfIdfDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.values.astype("int64")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32), torch.tensor(self.y[idx])

class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

def run():
    tr = pd.read_csv(TRAIN)
    va = pd.read_csv(VAL)

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=80000, min_df=2)
    Xtr = vec.fit_transform(tr["text"])
    Xva = vec.transform(va["text"])
    joblib.dump(vec, os.path.join(OUT, "kannada_lstm_tfidf.pkl"))

    dtr = TfIdfDataset(Xtr, tr["label"])
    dva = TfIdfDataset(Xva, va["label"])

    tr_loader = DataLoader(dtr, batch_size=256, shuffle=True)
    va_loader = DataLoader(dva, batch_size=512)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMClassifier(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(6):
        model.train()
        loop = tqdm(tr_loader, desc=f"Epoch {epoch+1}/6")
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            loop.set_postfix(loss=loss.item())

    # Validation
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(device)
            out = model(xb).cpu()
            preds.extend(out.argmax(1).tolist())
            true.extend(yb.tolist())

    rep = classification_report(true, preds, output_dict=True)
    print("\n[Kannada LSTM] Validation Report:\n")
    print(classification_report(true, preds))

    with open(os.path.join(OUT, "kannada_lstm_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    torch.save(model.state_dict(), os.path.join(OUT, "kannada_lstm.pt"))
    print("✔ Kannada LSTM model saved to", OUT)

if __name__ == "__main__":
    run()
