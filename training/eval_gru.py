import os, json, torch, joblib, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn

class SimpleGRUClassifier(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)


def evaluate_gru(model_path, vectorizer_path, test_csv, out_dir="evaluation/gru"):
    os.makedirs(out_dir, exist_ok=True)

    print("[INFO] Loading GRU model + vectorizer...")
    vec = joblib.load(vectorizer_path)

    input_dim = len(vec.get_feature_names_out())    # ⭐ FIXED

    model = SimpleGRUClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    test = pd.read_csv(test_csv)
    X = vec.transform(test["text"])
    y_true = test["label"].tolist()

    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
    y_pred = model(X_tensor).argmax(1).tolist()

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Oranges")
    plt.title("Confusion Matrix - GRU")
    plt.savefig(os.path.join(out_dir, "gru_confusion_matrix.png"))
    plt.close()

    rep = classification_report(y_true, y_pred, output_dict=True)
    with open(os.path.join(out_dir, "gru_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    pd.DataFrame(rep).transpose().plot(kind="barh", figsize=(10,6))
    plt.savefig(os.path.join(out_dir, "gru_report_plot.png"))
    plt.close()

    print("\n✅ GRU evaluation completed. Saved in:", out_dir)


if __name__ == "__main__":
    evaluate_gru(
        "models/deep_learning/gru_model.pt",
        "models/deep_learning/tfidf_dl.pkl",
        "data/labeled/test.csv"
    )
