# training/eval_xlmr.py
import os, json, torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification


def evaluate_xlmr(model_dir, test_csv, out_dir="evaluation/xlmr"):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using Device: {device}")

    print("[INFO] Loading XLM-R base architecture...")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=2
    ).to(device)

    trained_path = os.path.join(model_dir, "xlmr_model.pt")
    print("[INFO] Loading fine-tuned weights:", trained_path)
    model.load_state_dict(torch.load(trained_path, map_location=device))

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    # Load test CSV
    test = pd.read_csv(test_csv)

    # Ensure label is numeric (0 and 1)
    if test["label"].dtype == object:
        print("[INFO] Converting string labels to numeric classes...")
        label_map = {lbl: i for i, lbl in enumerate(test["label"].unique())}
        test["label"] = test["label"].map(label_map)
        print("Label map:", label_map)

    y_true = test["label"].tolist()
    y_pred = []

    print("[INFO] Predicting test samples...")
    for text in test["text"]:
        inputs = tokenizer.encode_plus(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(device)

        outputs = model(**inputs)
        pred = outputs.logits.argmax(1).item()
        y_pred.append(pred)

    # -------- Confusion Matrix --------
    print("[INFO] Saving confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("XLM-R Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "xlmr_confusion_matrix.png"))
    plt.close()

    # -------- Classification Report (JSON + Bar Plot) --------
    print("[INFO] Saving classification report...")
    rep = classification_report(y_true, y_pred, output_dict=True)

    with open(os.path.join(out_dir, "xlmr_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    pd.DataFrame(rep).transpose().plot(kind="barh", figsize=(10, 6))
    plt.title("XLM-R Classification Report")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "xlmr_report_plot.png"))
    plt.close()

    print("\n✅ XLM-R Evaluation Completed!")
    print("📁 Outputs saved in:", out_dir)


if __name__ == "__main__":
    evaluate_xlmr("models/transformer", "data/labeled/test.csv")
