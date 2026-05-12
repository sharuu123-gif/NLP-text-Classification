"""
evaluate.py
Model evaluation: accuracy, classification report, confusion matrix.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from config import BATCH_SIZE, LABEL_NAMES, PLOT_DIR, REPORT_DIR


def evaluate_model(model, test_dataset, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            outputs = model(**batch)
            preds   = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc    = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=list(LABEL_NAMES.values())
    )
    print(f"\n🎯 Accuracy: {acc:.4f}")
    print("\n📋 Classification Report:")
    print(report)

    # Save report
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES.values(),
                yticklabels=LABEL_NAMES.values())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.show()
    print(f"✅ Confusion matrix saved → {PLOT_DIR}")

    return acc, report
