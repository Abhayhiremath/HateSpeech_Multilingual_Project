import pandas as pd
import os
import re

RAW_DIR = "data/raw/new_languages"
OUT_DIR = "data/processed"

def clean_text(t):
    t = str(t)
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"[^A-Za-z0-9\u0600-\u06FF\u0C80-\u0CFF ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def preprocess_file(filename):
    print("\n=====================================")
    print(f"🔍 Processing: {filename}")
    print("=====================================")

    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)

    # auto-detect columns
    text_col = "text"
    label_col = "label"

    df = df[[text_col, label_col]]
    df[text_col] = df[text_col].apply(clean_text)
    df[label_col] = df[label_col].astype(int)

    lang = filename.split("_")[0].lower()
    out_path = os.path.join(OUT_DIR, f"{lang}_clean.csv")
    df.to_csv(out_path, index=False)

    print(f"✅ Saved cleaned dataset → {out_path}")
    print(f"📊 Total samples: {len(df)}")

if __name__ == "__main__":
    print("🚀 Starting preprocessing for all languages...")

    for f in os.listdir(RAW_DIR):
        if f.lower().endswith(".csv"):
            preprocess_file(f)

    print("\n🎉 Preprocessing finished!")
