"""
preprocess.py
Text cleaning, tokenization, and saving processed data.
"""

import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import RAW_DIR, PROCESSED_DIR

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Full preprocessing pipeline for a single text string."""
    text = str(text).lower()
    text = re.sub(r'<.*?>',        '',  text)   # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # Keep only letters
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w)
              for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_text to a DataFrame's 'text' column."""
    print("⏳ Preprocessing text...")
    df = df.copy()
    df['clean_text']   = df['text'].apply(clean_text)
    df['text_length']  = df['text'].apply(lambda x: len(x.split()))
    df['clean_length'] = df['clean_text'].apply(lambda x: len(x.split()))
    print("✅ Preprocessing complete!")
    return df


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save cleaned DataFrames to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train_clean.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR,  'test_clean.csv'),  index=False)
    print(f"💾 Saved → {PROCESSED_DIR}")


def load_processed():
    """Load previously saved cleaned CSVs."""
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_clean.csv'))
    test_df  = pd.read_csv(os.path.join(PROCESSED_DIR, 'test_clean.csv'))
    return train_df, test_df


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(RAW_DIR, 'test.csv'))
    train_df = preprocess_dataframe(train_df)
    test_df  = preprocess_dataframe(test_df)
    save_processed(train_df, test_df)
