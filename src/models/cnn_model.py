"""
cnn_model.py
TextCNN classifier using multiple kernel sizes.
"""

import torch
import torch.nn as nn
from config import VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, NUM_FILTERS


class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size=VOCAB_SIZE,
                 embed_dim=EMBED_DIM,
                 num_classes=NUM_CLASSES,
                 num_filters=NUM_FILTERS,
                 kernel_sizes=(2, 3, 4),
                 dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)          # (B, E, T)
        pooled = [
            torch.relu(conv(x)).max(dim=2)[0]           # (B, F)
            for conv in self.convs
        ]
        x = torch.cat(pooled, dim=1)                   # (B, F*K)
        return self.fc(self.dropout(x))


if __name__ == "__main__":
    model = TextCNN()
    print(model)
    dummy = torch.randint(0, VOCAB_SIZE, (8, 128))
    print("Output shape:", model(dummy).shape)
