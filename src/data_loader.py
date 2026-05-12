"""
data_loader.py
Loads AG News dataset from HuggingFace and saves locally as CSV.
"""

import os
import pandas as pd
from datasets import load_dataset
from config import RAW_DIR, LABEL_NAMES


def load_ag_news():
    """Download and return AG News as pandas DataFrames."""
    print("⏳ Loading AG News dataset from HuggingFace...")
    dataset = load_dataset("ag_news")

    train_df = pd.DataFrame(dataset['train'])
    test_df  = pd.DataFrame(dataset['test'])

    # Map numeric labels to category names
    train_df['category'] = train_df['label'].map(LABEL_NAMES)
    test_df['category']  = test_df['label'].map(LABEL_NAMES)

    print(f"✅ Train: {train_df.shape[0]:,} rows")
    print(f"✅ Test : {test_df.shape[0]:,} rows")
    return train_df, test_df


def save_raw(train_df, test_df):
    """Save raw DataFrames to data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(RAW_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(RAW_DIR,  'test.csv'),  index=False)
    print(f"💾 Saved → {RAW_DIR}")


def load_raw():
    """Load previously saved raw CSVs."""
    train_df = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(RAW_DIR, 'test.csv'))
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_ag_news()
    save_raw(train_df, test_df)
