import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Input dataset (your full labeled data)
INPUT_CSV = "data/labeled/final_dataset.csv"   # <- change this to your full file path

# Output folder
OUT_DIR = "data/splits"
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_CSV)

print("[INFO] Total Samples:", len(df))

# STEP 1: Split into Train (80%) + Test (20%)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

# STEP 2: Split Train into Train (90%) + Val (10%)
train_df, val_df = train_test_split(train_df, test_size=0.10, random_state=42, shuffle=True)

# Save files
train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

print("\n✅ Splits created and saved!")
print("Training samples :", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples     :", len(test_df))
print(f"\nFiles saved in: {OUT_DIR}/")
