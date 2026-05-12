"""
train.py
Training loop for BERT (and easily adaptable for LSTM/CNN).
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_DIR, BERT_MODEL_NAME


def train_bert(train_dataset, num_classes=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_classes)
    model.to(device)

    loader    = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch+1}/{EPOCHS}  |  Avg Loss: {avg_loss:.4f}")

    # Save model
    save_path = os.path.join(MODEL_DIR, 'bert_model')
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"💾 Model saved → {save_path}")
    return model
