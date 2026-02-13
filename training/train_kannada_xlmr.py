# training/train_kannada_xlmr_high_accuracy.py

import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)

TRAIN = r"D:\HateSpeech_Multilingual_Project\data\kannada\train.csv"
VAL   = r"D:\HateSpeech_Multilingual_Project\data\kannada\val.csv"
OUT   = r"D:\HateSpeech_Multilingual_Project\models\kannada"
os.makedirs(OUT, exist_ok=True)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            str(self.texts[idx]),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        loss = out.loss
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        pred = out.logits.argmax(1).cpu().tolist()
        lab = batch["labels"].cpu().tolist()
        preds.extend(pred)
        labels.extend(lab)
    return labels, preds

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    tr = pd.read_csv(TRAIN)
    va = pd.read_csv(VAL)

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    ds_train = TextDataset(tr["text"], tr["label"], tokenizer)
    ds_val   = TextDataset(va["text"], va["label"], tokenizer)

    # Balanced sampler
    class_counts = tr["label"].value_counts().to_dict()
    weights = [1.0 / class_counts[l] for l in tr["label"]]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(ds_train, batch_size=8, sampler=sampler)
    val_loader   = DataLoader(ds_val, batch_size=16, shuffle=False)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    EPOCHS = 6
    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for ep in range(EPOCHS):
        print(f"\n===== Epoch {ep+1}/{EPOCHS} =====")
        loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"[INFO] Train Loss: {loss:.4f}")

        y_true, y_pred = eval_epoch(model, val_loader, device)
        print(classification_report(y_true, y_pred))

        report = classification_report(y_true, y_pred, output_dict=True)
        with open(os.path.join(OUT, f"kannada_xlmr_epoch{ep+1}.json"), "w") as f:
            json.dump(report, f, indent=2)

    save_path = os.path.join(OUT, "kannada_xlmr_high_accuracy.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 Kannada XLM-R high-accuracy model saved → {save_path}")

if __name__ == "__main__":
    run()
