"""
bert_model.py
Fine-tuning BERT for sequence classification.
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from config import BERT_MODEL_NAME, NUM_CLASSES, MAX_LEN


def get_bert_model(num_classes=NUM_CLASSES):
    """Load pre-trained BERT and add a classification head."""
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=num_classes
    )
    return model


def predict_single(text: str, model, tokenizer, device='cpu') -> dict:
    """Run inference on a single string."""
    label_names = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LEN,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = outputs.logits.argmax(dim=1).item()
    return {'label': pred_id, 'category': label_names[pred_id]}


if __name__ == "__main__":
    model     = get_bert_model()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    sample    = "Apple announces new MacBook with M3 chip at WWDC event"
    result    = predict_single(sample, model, tokenizer)
    print("Prediction:", result)
