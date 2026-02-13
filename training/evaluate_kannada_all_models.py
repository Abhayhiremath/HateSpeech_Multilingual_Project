# training/evaluate_kannada_all_models.py

import os
import json
import joblib
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# =====================================
# PATHS
# =====================================
TEST_CSV = r"D:\HateSpeech_Multilingual_Project\data\kannada\test.csv"
MODEL_DIR = r"D:\HateSpeech_Multilingual_Project\models\kannada"
OUT_DIR = r"D:\HateSpeech_Multilingual_Project\evaluation\kannada"

os.makedirs(OUT_DIR, exist_ok=True)


# =====================================
# Helper: Save confusion matrix
# =====================================
def save_conf_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()


# =====================================
# 1. Evaluate SVM & Naive Bayes
# =====================================
def eval_svm_nb(test_df):
    results = {}

    for name, vec_file, model_file in [
        ("svm", "kannada_svm_vectorizer.pkl", "kannada_svm.pkl"),
        ("nb", "kannada_nb_vectorizer.pkl", "kannada_nb.pkl"),
    ]:
        print(f"\n===== Evaluating {name.upper()} =====")

        vec = joblib.load(os.path.join(MODEL_DIR, vec_file))
        clf = joblib.load(os.path.join(MODEL_DIR, model_file))

        X = vec.transform(test_df["text"])
        y_true = test_df["label"]
        y_pred = clf.predict(X)

        rep = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))

        # save report
        with open(os.path.join(OUT_DIR, f"{name}_report.json"), "w") as f:
            json.dump(rep, f, indent=2)

        # save confusion matrix
        save_conf_matrix(y_true, y_pred, f"{name.upper()} Confusion Matrix", f"{name}_confusion.png")

        results[name] = rep

    return results


# =====================================
# 2. Evaluate LSTM & GRU (TF-IDF classifiers)
# =====================================
class TfIdfDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.values.astype("int64")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx].toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32), torch.tensor(self.y[idx])


def eval_lstm_gru(test_df):
    results = {}

    for name, vec_file, model_file in [
        ("lstm", "kannada_lstm_tfidf.pkl", "kannada_lstm.pt"),
        ("gru",  "kannada_gru_tfidf.pkl",  "kannada_gru.pt")
    ]:
        print(f"\n===== Evaluating {name.upper()} =====")

        vec = joblib.load(os.path.join(MODEL_DIR, vec_file))
        X = vec.transform(test_df["text"])
        ds = TfIdfDataset(X, test_df["label"])
        loader = DataLoader(ds, batch_size=128)

        # load model
        input_dim = X.shape[1]

        class SimpleNN(torch.nn.Module):
            def __init__(self, in_dim, hidden=256):
                super().__init__()
                self.fc1 = torch.nn.Linear(in_dim, hidden)
                self.act = torch.nn.ReLU()
                self.drop = torch.nn.Dropout(0.3)
                self.fc2 = torch.nn.Linear(hidden, 2)

            def forward(self, x):
                return self.fc2(self.drop(self.act(self.fc1(x))))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleNN(input_dim).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_file), map_location=device))
        model.eval()

        preds, true = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                out = model(xb).cpu()
                preds.extend(out.argmax(1).tolist())
                true.extend(yb.tolist())

        rep = classification_report(true, preds, output_dict=True)
        print(classification_report(true, preds))

        with open(os.path.join(OUT_DIR, f"{name}_report.json"), "w") as f:
            json.dump(rep, f, indent=2)

        save_conf_matrix(true, preds, f"{name.upper()} Confusion Matrix", f"{name}_confusion.png")

        results[name] = rep

    return results


# =====================================
# 3. Evaluate XLM-R (Transformer)
# =====================================
def eval_xlmr(test_df):
    print("\n===== Evaluating XLM-R Transformer =====")

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    class XLMDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = list(labels)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    ds = XLMDataset(test_df["text"], test_df["label"])
    loader = DataLoader(ds, batch_size=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2
    ).to(device)

    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "kannada_xlmr_high_accuracy.pt"),
        map_location=device
    ))

    preds, true = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds.extend(out.logits.argmax(1).cpu().tolist())
            true.extend(batch["labels"].cpu().tolist())

    rep = classification_report(true, preds, output_dict=True)
    print(classification_report(true, preds))

    with open(os.path.join(OUT_DIR, f"xlmr_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    save_conf_matrix(true, preds, "XLM-R Confusion Matrix", "xlmr_confusion.png")

    return rep


# =====================================
# MAIN
# =====================================
def main():
    test_df = pd.read_csv(TEST_CSV)
    print("\nLoaded Test Dataset:", test_df.shape)

    results = {}
    results["svm_nb"] = eval_svm_nb(test_df)
    results["lstm_gru"] = eval_lstm_gru(test_df)
    results["xlmr"]     = eval_xlmr(test_df)

    print("\n✔ All 5 Kannada models evaluated successfully!")
    print("Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
