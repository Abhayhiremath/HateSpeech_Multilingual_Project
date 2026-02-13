# prepare_data.py
import os
from preprocessing.utils import load_many, clean_frame, train_val_test_split

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned"
LAB_DIR = "data/labeled"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(LAB_DIR, exist_ok=True)

print("=======================================")
print("        DATA PREPARATION START          ")
print("=======================================")

# -----------------------------------------
# LOAD ALL 8 CSV DATASETS
# -----------------------------------------
print("[1] Loading datasets from:", RAW_DIR)
df = load_many(RAW_DIR)
print(f"[INFO] Loaded total rows: {len(df)}")

# -----------------------------------------
# CLEAN + NORMALIZE
# -----------------------------------------
print("[2] Cleaning & normalizing...")
df = clean_frame(df)
print(f"[INFO] After cleaning: {len(df)} rows")

# Save cleaned master
clean_path = os.path.join(CLEAN_DIR, "cleaned_master.csv")
df.to_csv(clean_path, index=False)
print(f"[INFO] Saved cleaned_master.csv → {clean_path}")

# -----------------------------------------
# SPLIT DATA
# -----------------------------------------
print("[3] Splitting into train/val/test...")
train, val, test = train_val_test_split(df)

train.to_csv(os.path.join(LAB_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(LAB_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(LAB_DIR, "test.csv"), index=False)

print(f"[DONE] Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print("=======================================")
print("        DATA PREPARATION DONE           ")
print("=======================================")
