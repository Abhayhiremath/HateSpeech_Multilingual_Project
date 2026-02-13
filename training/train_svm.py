# training/train_svm.py
import argparse, os, joblib, json, time, psutil
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

# -------------------------
# Helper: show status
# -------------------------
def status(msg):
    print(f"\n✅ {msg}")

def warn(msg):
    print(f"\n⚠️  {msg}")

def info(msg):
    print(f"\n[INFO] {msg}")

# -------------------------
# TRAINING FUNCTION
# -------------------------
def build_and_train(model_type: str, train_csv: str, val_csv: str, out_dir: str):
    start_total = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # LOAD DATA
    # -------------------------
    info("Loading datasets...")
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    print(f"Training samples: {len(train)}  |  Validation samples: {len(val)}")

    # -------------------------
    # TF–IDF
    # -------------------------
    info("Initializing TF-IDF...")
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2
    )

    info("Fitting TF-IDF (live progress)...")
    tfidf_start = time.time()
    X_train = vec.fit_transform(tqdm(train["text"], desc="Building Vocabulary"))
    tfidf_end = time.time()

    status(f"TF-IDF completed in {tfidf_end - tfidf_start:.2f} sec")
    status(f"TF-IDF Matrix Shape: {X_train.shape}")

    # -------------------------
    # VALIDATION TRANSFORM
    # -------------------------
    info("Transforming validation set...")
    X_val = vec.transform(val["text"])

    # -------------------------
    # MODEL SELECTION
    # -------------------------
    if model_type == "svm":
        clf = LinearSVC()
        model_name = "svm.pkl"
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=15000, n_jobs=-1)
        model_name = "logistic.pkl"

    # -------------------------
    # LIVE TRAINING SIMULATION
    # (SVM/LogReg train instantly; we simulate visible progress)
    # -------------------------
    info(f"Training model: {model_type.upper()}...")

    simulated_steps = 20
    for i in tqdm(range(simulated_steps), desc="Training Progress"):
        time.sleep(0.05)

    train_start = time.time()
    clf.fit(X_train, train["label"])
    train_end = time.time()

    status(f"Model training finished in {train_end - train_start:.2f} sec")

    # -------------------------
    # VALIDATION PREDICTIONS
    # -------------------------
    info("Predicting validation batches...")
    preds = []
    n_val = X_val.shape[0]

    for i in tqdm(range(0, n_val, 4000), desc="Predicting"):
        batch = X_val[i: i+4000]
        batch_pred = clf.predict(batch)
        preds.extend(batch_pred)

    preds = np.array(preds)

    # -------------------------
    # REPORT
    # -------------------------
    info("Generating classification report...")
    report = classification_report(val["label"], preds)
    print(report)

    with open(os.path.join(out_dir, f"{model_type}_val_report.json"), "w") as f:
        json.dump(classification_report(val["label"], preds, output_dict=True), f, indent=2)

    # -------------------------
    # SAVE MODEL & VECTORIZER
    # -------------------------
    info("Saving model + vectorizer...")
    joblib.dump(clf, os.path.join(out_dir, model_name))
    joblib.dump(vec, os.path.join(out_dir, "vectorizer.pkl"))

    # -------------------------
    # FINAL SUMMARY
    # -------------------------
    total_time = time.time() - start_total
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    status("TRAINING COMPLETE ✅")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Memory Used: {mem:.2f} MB")
    print(f"Model Saved: {os.path.join(out_dir, model_name)}")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="models/traditional")
    ap.add_argument("--type", choices=["svm", "logreg"], required=True)
    args = ap.parse_args()

    build_and_train(args.type, args.train, args.val, args.out)

