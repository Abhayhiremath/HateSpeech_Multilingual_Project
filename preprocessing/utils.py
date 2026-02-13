# preprocessing/utils.py
import os
import glob
import pandas as pd
from .clean_text import basic_clean
from .remove_duplicates import dedupe
from .label_builder import unify_binary

# --------------------------------------------------------
# TEXT + LABEL COLUMN SUPPORT (BASED ON YOUR ACTUAL FILES)
# --------------------------------------------------------
TEXT_COLUMNS = [
    "Text",                 # your main column (capital T)
    "text",
    "comment_text",         # kaggle clean
    "tweet",                # twitter datasets
    "clean_text",
    "content",
    "body",
    "message"
]

LABEL_COLUMNS = [
    "oh_label",             # all your parsed datasets use this
    "label",
    "toxic",
    "hate"
]

# --------------------------------------------------------
# DETECT TEXT COLUMN
# --------------------------------------------------------
def find_text_column(df):
    # Priority: known text columns
    for col in TEXT_COLUMNS:
        if col in df.columns:
            return col

    # Fallback: pick column with longest average text
    text_like = sorted(
        df.columns,
        key=lambda c: df[c].astype(str).str.len().mean(),
        reverse=True
    )
    return text_like[0]

# --------------------------------------------------------
# DETECT LABEL COLUMN
# --------------------------------------------------------
def find_label_column(df):
    for col in LABEL_COLUMNS:
        if col in df.columns:
            return col

    # No label found → default = 0
    df["label"] = 0
    return "label"

# --------------------------------------------------------
# MAIN LOADER
# --------------------------------------------------------
def load_many(csv_dir: str) -> pd.DataFrame:
    frames = []
    files = glob.glob(os.path.join(csv_dir, "*.csv"))
    print("\n[INFO] Found files:", files)

    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"[OK] Loaded {os.path.basename(f)} with columns: {df.columns.tolist()}")

            text_col = find_text_column(df)
            label_col = find_label_column(df)

            print(f"     → text column   = {text_col}")
            print(f"     → label column  = {label_col}")

            df = df[[text_col, label_col]].rename(
                columns={text_col: "text", label_col: "label"}
            )

            # Ensure binary labels
            df["label"] = df["label"].astype(int).clip(0, 1)

            frames.append(df)

        except Exception as e:
            print(f"[SKIP] {f} → {e}")

    if not frames:
        raise ValueError("No usable CSVs found!")

    return pd.concat(frames, ignore_index=True)

# --------------------------------------------------------
# CLEAN FRAME
# --------------------------------------------------------
def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Cleaning text...")
    df["text"] = df["text"].astype(str).map(basic_clean)

    print("[INFO] Removing duplicates...")
    df = dedupe(df, "text")

    print("[INFO] Normalizing labels...")
    df = unify_binary(df, "text", "label")

    df = df[df["text"].str.len() >= 3]
    return df

# --------------------------------------------------------
# SPLIT
# --------------------------------------------------------
def train_val_test_split(df, seed=42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    train_end = int(0.8 * n)
    val_end   = int(0.9 * n)

    train = df[:train_end]
    val   = df[train_end:val_end]
    test  = df[val_end:]

    return train, val, test
