# training/train_nb.py
import argparse, os, joblib, json, time, psutil
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np

# -------------------------
# Helper printing
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
def train_nb(train_csv: str, val_csv: str, out_dir: str):

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
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2
    )

    info("Fitting TF-IDF (progress below)...")
    X_train = vectorizer.fit_transform(tqdm(train["text"], desc="Building Vocabulary"))
    X_val = vectorizer.transform(val["text"])

    status(f"TF-IDF Completed — Shape: {X_train.shape}")

    # -------------------------
    # TRAIN NAIVE BAYES
    # -------------------------
    info("Training Naive Bayes model (MultinomialNB)...")

    nb = MultinomialNB()

    start_train = time.time()
    nb.fit(X_train, train["label"])
    end_train = time.time()

    status(f"NB Training completed in {end_train - start_train:.2f} sec")

    # -------------------------
    # VALIDATION
    # -------------------------
    info("Predicting validation dataset...")
    preds = nb.predict(X_val)

    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------
    info("Generating classification report...")
    report = classification_report(val["label"], preds)
    print(report)

    with open(os.path.join(out_dir, "nb_val_report.json"), "w") as f:
        json.dump(
            classification_report(val["label"], preds, output_dict=True),
            f, indent=2
        )

    # -------------------------
    # SAVE MODEL + VECTORIZER
    # -------------------------
    info("Saving model & vectorizer...")
    joblib.dump(nb, os.path.join(out_dir, "nb.pkl"))
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.pkl"))

    # -------------------------
    # SUMMARY
    # -------------------------
    total_time = time.time() - start_total
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    status("TRAINING COMPLETE ✅")
    print(f"Total Time: {total_time:.2f} sec")
    print(f"Memory Used: {mem:.2f} MB")
    print(f"Saved: {os.path.join(out_dir, 'nb.pkl')}")


# -----------------------------------------------------------
# ✅ THIS WAS MISSING — MAIN BLOCK (NOW ADDED!)
# -----------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="models/traditional")
    args = ap.parse_args()

    train_nb(
        train_csv=args.train,
        val_csv=args.val,
        out_dir=args.out
    )
