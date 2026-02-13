# training/split_kannada_merged.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = r"D:\HateSpeech_Multilingual_Project\data\kannada\kannada_merged_600.csv"
OUT_DIR = r"D:\HateSpeech_Multilingual_Project\data\kannada"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

# 70% train, 30% temp
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"]
)

# 15% val, 15% test (split 30% into half)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print("✔ Kannada merged dataset split complete!")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)
