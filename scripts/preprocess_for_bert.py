import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# Input/output paths
RAW_PATH = "data/raw/train.csv"
PROCESSED_PATH = "data/processed/train_clean.csv"

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove non-alphabetic characters
    text = text.lower().strip()
    return text

def preprocess():
    df = pd.read_csv(RAW_PATH)
    df = df[["text", "target"]]
    df.dropna(inplace=True)
    df["text"] = df["text"].apply(clean_text)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Cleaned data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()
