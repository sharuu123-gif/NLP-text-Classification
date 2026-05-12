"""
feature_engineering.py
TF-IDF, Word2Vec embeddings, and BERT tokenization.
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from config import MAX_LEN, BERT_MODEL_NAME, VOCAB_SIZE


# ── TF-IDF ─────────────────────────────────────────────────────────────────────

def build_tfidf(train_texts, test_texts, max_features=VOCAB_SIZE):
    """Fit TF-IDF on train and transform both splits."""
    print("⏳ Building TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_texts)
    X_test  = tfidf.transform(test_texts)
    print(f"✅ TF-IDF shape — Train: {X_train.shape}, Test: {X_test.shape}")
    return tfidf, X_train, X_test


# ── BERT Tokenizer ─────────────────────────────────────────────────────────────

def bert_tokenize(texts, max_len=MAX_LEN):
    """Tokenize a list of strings with BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    encodings = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    print(f"✅ Tokenized {len(texts):,} samples")
    return encodings


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
