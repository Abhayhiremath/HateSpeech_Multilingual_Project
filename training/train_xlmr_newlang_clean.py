# training/train_xlmr_newlang_clean.py

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

###############################################
# DATASET
###############################################

class XLMRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx]).long(),
        }

###############################################
# TRAIN FUNCTION
###############################################

def train_xlmr(lang):

    input_file = f"data/processed/{lang}_clean.csv"
    print(f"\n📂 Loading dataset for {lang}...")
    df = pd.read_csv(input_file)
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    save_dir = f"models/{lang}_models"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    train_ds = XLMRDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_ds  = XLMRDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on {device.upper()}")

    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_accuracy = 0

    print("\n📘 Starting fine-tuning XLM-R...\n")

    for epoch in range(3):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss = {total_loss/len(train_loader):.4f}")

        # ---------------- VALIDATION -----------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids=ids, attention_mask=mask).logits
                _, preds = torch.max(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Validation Accuracy = {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            # FULL model save (fixed)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            print(f"💾 Saved best model at {save_dir}\n")


###############################################
# MAIN
###############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    train_xlmr(args.lang)
