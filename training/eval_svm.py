import os, json, joblib, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_svm(model_path, test_csv, out_dir="evaluation/svm"):
    os.makedirs(out_dir, exist_ok=True)
    print("[INFO] Loading SVM model + vectorizer...")
    model = joblib.load(model_path)
    vec = joblib.load(os.path.join(os.path.dirname(model_path), "vectorizer.pkl"))
    
    test = pd.read_csv(test_csv)
    X_test = vec.transform(test["text"])
    y_true = test["label"]

    print("[INFO] Predicting...")
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.savefig(os.path.join(out_dir, "svm_confusion_matrix.png"))
    plt.close()

    rep = classification_report(y_true, y_pred, output_dict=True)
    with open(os.path.join(out_dir, "svm_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    pd.DataFrame(rep).transpose().plot(kind="barh")
    plt.savefig(os.path.join(out_dir, "svm_report_plot.png"))
    plt.close()

    print("\n✅ SVM evaluation completed:", out_dir)

if __name__ == "__main__":
    evaluate_svm("models/traditional/svm.pkl", "data/labeled/test.csv")
