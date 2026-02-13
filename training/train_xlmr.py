# training/train_xlmr.py

import argparse
import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW   # ✅ FIXED: use torch.optim AdamW


# --------------------------
# Dataset
# --------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            str(self.texts[idx]),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# --------------------------
# Training Loop
# --------------------------
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, labels = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        pred = outputs.logits.argmax(dim=1).cpu().tolist()
        lab = batch["labels"].cpu().tolist()

        preds.extend(pred)
        labels.extend(lab)

    return labels, preds


# --------------------------
# Main Training Function
# --------------------------
def run(train_csv, val_csv, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    train_ds = TextDataset(train_df["text"], train_df["label"], tokenizer)
    val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    total_steps = len(train_loader) * 2  # 2 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # --------------------------
    # TRAINING
    # --------------------------
    for epoch in range(2):
        print(f"\n===== Epoch {epoch+1}/2 =====")

        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"[INFO] Train Loss: {loss:.4f}")

        y_true, y_pred = eval_epoch(model, val_loader, device)

        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))

        with open(os.path.join(out_dir, f"xlmr_epoch{epoch+1}_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    # --------------------------
    # SAVE FINAL MODEL
    # --------------------------
    save_path = os.path.join(out_dir, "xlmr_model.pt")
    torch.save(model.state_dict(), save_path)

    print(f"\n🎉 Training Complete! Saved model → {save_path}")


# --------------------------
# CLI Entry
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="models/transformer")
    args = ap.parse_args()

    run(args.train, args.val, args.out)
