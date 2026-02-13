# training/evaluate_all_new.py

import os
import argparse
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#######################################################
# LSTM & GRU MODELS (TF-IDF based)
#######################################################

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


#######################################################
# XLM-R DATASET
#######################################################

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


#######################################################
# CONFUSION MATRIX PLOT (PNG)
#######################################################

def save_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


#######################################################
# MAIN EVALUATION FUNCTION
#######################################################

def evaluate_all(lang):

    print(f"\n===== Evaluating Models for {lang.upper()} =====\n")

    # Load dataset & split
    df = pd.read_csv(f"data/processed/{lang}_clean.csv")
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    model_dir = f"models/{lang}_models"

    # Paths (classical models)
    svm_path    = os.path.join(model_dir, "svm.pkl")
    logreg_path = os.path.join(model_dir, "logistic.pkl")

    # TF-IDF vectorizers for deep models
    tfidf_lstm = joblib.load(os.path.join(model_dir, "tfidf_lstm.pkl"))
    tfidf_gru  = joblib.load(os.path.join(model_dir, "tfidf_gru.pkl"))

    ###################################################
    # 1. SVM  (Pipeline: raw text → prediction)
    ###################################################
    print("\nEvaluating SVM...")

    svm_model = joblib.load(svm_path)
    svm_preds = svm_model.predict(X_test)

    print(classification_report(y_test, svm_preds, digits=4))
    save_confusion_matrix(
        confusion_matrix(y_test, svm_preds),
        ["Non-Hate", "Hate"],
        os.path.join(model_dir, "svm_confusion_new.png")
    )

    ###################################################
    # 2. Logistic Regression (Pipeline)
    ###################################################
    print("\nEvaluating Logistic Regression...")

    logreg = joblib.load(logreg_path)
    logreg_preds = logreg.predict(X_test)

    print(classification_report(y_test, logreg_preds, digits=4))
    save_confusion_matrix(
        confusion_matrix(y_test, logreg_preds),
        ["Non-Hate", "Hate"],
        os.path.join(model_dir, "logistic_confusion_new.png")
    )

    ###################################################
    # 3. LSTM (TF-IDF → LSTMClassifier)
    ###################################################
    print("\nEvaluating LSTM...")

    input_dim_lstm = tfidf_lstm.transform([X_test[0]]).shape[1]
    lstm = LSTMClassifier(input_dim_lstm)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, "lstm_best.pt"), map_location="cpu"))
    lstm.eval()

    lstm_preds = []
    for text in X_test:
        vec = tfidf_lstm.transform([text]).toarray()
        x = torch.tensor(vec).float().unsqueeze(0)  # [1, 1, input_dim]
        out = lstm(x)
        lstm_preds.append(out.argmax(1).item())

    print(classification_report(y_test, lstm_preds, digits=4))
    save_confusion_matrix(
        confusion_matrix(y_test, lstm_preds),
        ["Non-Hate", "Hate"],
        os.path.join(model_dir, "lstm_confusion_new.png")
    )

    ###################################################
    # 4. GRU (TF-IDF → GRUClassifier)
    ###################################################
    print("\nEvaluating GRU...")

    input_dim_gru = tfidf_gru.transform([X_test[0]]).shape[1]
    gru = GRUClassifier(input_dim_gru)
    gru.load_state_dict(torch.load(os.path.join(model_dir, "gru_best.pt"), map_location="cpu"))
    gru.eval()

    gru_preds = []
    for text in X_test:
        vec = tfidf_gru.transform([text]).toarray()
        x = torch.tensor(vec).float().unsqueeze(0)  # [1, 1, input_dim]
        out = gru(x)
        gru_preds.append(out.argmax(1).item())

    print(classification_report(y_test, gru_preds, digits=4))
    save_confusion_matrix(
        confusion_matrix(y_test, gru_preds),
        ["Non-Hate", "Hate"],
        os.path.join(model_dir, "gru_confusion_new.png")
    )

    ###################################################
    # 5. XLM-R (Full HF model from models/<lang>_models)
    ###################################################
    print("\nEvaluating XLM-R...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned model + tokenizer from your model_dir
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
    xlmr = XLMRobertaForSequenceClassification.from_pretrained(model_dir).to(device)

    test_ds = XLMRDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=8)

    xlmr.eval()
    xlmr_preds = []

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = xlmr(ids, attention_mask=mask).logits
            xlmr_preds.extend(logits.argmax(1).cpu().numpy())

    print(classification_report(y_test, xlmr_preds, digits=4))
    save_confusion_matrix(
        confusion_matrix(y_test, xlmr_preds),
        ["Non-Hate", "Hate"],
        os.path.join(model_dir, "xlmr_confusion_new.png")
    )

    print("\n🎉 All models evaluated successfully!\n")


#######################################################
# ENTRY POINT
#######################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Language code, e.g. 'arabic' or 'spanish'")
    args = parser.parse_args()

    evaluate_all(args.lang)
