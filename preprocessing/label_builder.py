# preprocessing/label_builder.py
import pandas as pd

# Expect input CSVs with at least: text,label (0/1). If not, map using heuristics here.
def unify_binary(df: pd.DataFrame, text_col="text", label_col="label") -> pd.DataFrame:
    # coerce to {0,1}
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]]
    df["label"] = df["label"].astype(int).clip(0, 1)
    return df
