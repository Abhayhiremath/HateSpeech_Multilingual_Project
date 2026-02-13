# training/train_svm_newlang.py

import pandas as pd
import os
import argparse
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

def train_svm(lang):
    input_file = f"data/processed/{lang}_clean.csv"

    print(f"\n[INFO] Loading dataset → {input_file}")
    df = pd.read_csv(input_file)
    print(f"[INFO] Loaded {len(df)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000)),
        ("svm", LinearSVC())
    ])

    print("\n🚀 Training SVM...")
    pipe.fit(X_train, y_train)
    print("✔ Training completed.")

    preds = pipe.predict(X_test)
    print("\n📊 Accuracy:", accuracy_score(y_test, preds))
    print("\n📄 Classification Report:")
    print(classification_report(y_test, preds))

    out_dir = f"models/{lang}_models"
    os.makedirs(out_dir, exist_ok=True)

    save_path = f"{out_dir}/svm.pkl"
    joblib.dump(pipe, save_path)
    print(f"\n💾 Saved SVM model → {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="arabic or spanish")
    args = parser.parse_args()
    train_svm(args.lang)
