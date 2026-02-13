# preprocessing/remove_duplicates.py
import pandas as pd

def dedupe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[text_col])
    df = df.drop_duplicates(subset=[text_col])
    print(f"[dedupe] {before} -> {len(df)} rows")
    return df
