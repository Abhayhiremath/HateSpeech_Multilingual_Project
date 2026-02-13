import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    # First split → Train (70%) & Temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )

    # Second split → Val (15%) & Test (15%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    # Save files
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    print("\n✔ SPLIT COMPLETE")
    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)

if __name__ == "__main__":
    # MODIFY ONLY THESE TWO LINES
    INPUT = r"D:\HateSpeech_Multilingual_Project\data\kannada\kannada_test_200.csv"
    OUT_DIR = r"D:\HateSpeech_Multilingual_Project\data\kannada"

    split_dataset(INPUT, OUT_DIR)
