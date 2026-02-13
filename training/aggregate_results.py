import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", "kannada")

MODELS = ["svm", "nb", "lstm", "gru", "xlmr"]

def load_report(model):
    path = os.path.join(EVAL_DIR, f"{model}_report.json")
    with open(path) as f:
        return json.load(f)

print("\nFINAL MODEL EVALUATION RESULTS (Kannada Dataset)\n")

results = {}

for model in MODELS:
    rep = load_report(model)
    weighted = rep["weighted avg"]

    results[model.upper()] = {
        "Precision": round(weighted["precision"], 4),
        "Recall": round(weighted["recall"], 4),
        "F1-Score": round(weighted["f1-score"], 4)
    }

for model, metrics in results.items():
    print(model, metrics)
