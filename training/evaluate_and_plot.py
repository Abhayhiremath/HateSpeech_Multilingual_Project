# training/evaluate_and_plot.py

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)

def load_model_and_vectorizer(model_path):
    """Load classifier + vectorizer if available."""
    model = joblib.load(model_path)

    vec_path = os.path.join(os.path.dirname(model_path), "vectorizer.pkl")
    if os.path.exists(vec_path):
        vectorizer = joblib.load(vec_path)
        print("[INFO] Loaded vectorizer.pkl")
    else:
        vectorizer = None

    return model, vectorizer


def safe_predict(model, vectorizer, X_raw):
    """Predict labels safely (handles SVM, LogReg, Pipelines)."""
    if vectorizer is not None:
        X = vectorizer.transform(X_raw)
    else:
        X = X_raw

    return model.predict(X), X


def safe_probabilities(model, vectorizer, X_raw, y_pred):
    """Generate probabilities or fallback."""
    if vectorizer is not None:
        X = vectorizer.transform(X_raw)
    else:
        X = X_raw

    # Predict_proba models
    try:
        return model.predict_proba(X)[:, 1]
    except:
        pass

    # Decision function
    try:
        df = model.decision_function(X)
        df = (df - df.min()) / (df.max() - df.min() + 1e-6)
        return df
    except:
        pass

    # Last fallback (for SVM)
    print("[WARN] No probability output. Using fallback.")
    return np.where(y_pred == 1, 0.8, 0.2)


def plot_confusion_matrix(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Harmful", "Harmful"],
                yticklabels=["Non-Harmful", "Harmful"])
    plt.title(title)
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_probs, out_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.title(title)
    plt.savefig(out_path)
    plt.close()


def plot_pr_curve(y_true, y_probs, out_path, title):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()


def plot_metric_bars(metrics, out_path, title):
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(6,5))
    plt.bar(names, values, color=["blue","orange","green","red"])
    plt.ylim(0,1)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()


def evaluate_model(model_path, test_csv, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)

    print("\n[INFO] Loading model + vectorizer...")
    model, vectorizer = load_model_and_vectorizer(model_path)

    test = pd.read_csv(test_csv)
    X_raw = test["text"]
    y_true = test["label"]

    print("[INFO] Predicting labels...")
    y_pred, X_vect = safe_predict(model, vectorizer, X_raw)

    print("[INFO] Getting probabilities...")
    y_prob = safe_probabilities(model, vectorizer, X_raw, y_pred)

    print("[INFO] Classification Report:")
    print(classification_report(y_true, y_pred))

    # Save metrics JSON
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Save plots
    plot_confusion_matrix(y_true, y_pred, os.path.join(out_dir, f"{name}_cm.png"),
                          f"{name.upper()} — Confusion Matrix")

    plot_roc_curve(y_true, y_prob, os.path.join(out_dir, f"{name}_roc.png"),
                   f"{name.upper()} — ROC Curve")

    plot_pr_curve(y_true, y_prob, os.path.join(out_dir, f"{name}_pr.png"),
                  f"{name.upper()} — PR Curve")

    plot_metric_bars(
        {
            "Accuracy": report["accuracy"],
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-Score": report["weighted avg"]["f1-score"],
        },
        os.path.join(out_dir, f"{name}_metrics.png"),
        f"{name.upper()} — Metrics"
    )

    print(f"\n✅ DONE! Images saved in: {out_dir}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="evaluation")
    args = ap.parse_args()

    evaluate_model(args.model, args.test, args.out, args.name)
