# training/evaluate.py
import argparse, json, os, joblib, pandas as pd
from sklearn.metrics import classification_report

def eval_sklearn(model_path, csv_path):
    pipe = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    pred = pipe.predict(df["text"])
    rep = classification_report(df["label"], pred, output_dict=True)
    print(classification_report(df["label"], pred))
    return rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="evaluation/metrics_report.json")
    args = ap.parse_args()
    rep = eval_sklearn(args.model, args.test)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f: json.dump(rep, f, indent=2)
