"""
lstm_model.py
Bidirectional LSTM classifier for text classification.
"""

import torch
import torch.nn as nn
from config import VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES


class LSTMClassifier(nn.Module):
    def __init__(self,
                 vocab_size=VOCAB_SIZE,
                 embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES,
                 num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, num_classes)  # *2 → bidirectional

    def forward(self, x):
        x = self.embedding(x)                          # (B, T, E)
        _, (hidden, _) = self.lstm(x)                 # hidden: (2*L, B, H)
        # Concat last forward & backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, 2H)
        out = self.fc(self.dropout(hidden))
        return out


if __name__ == "__main__":
    model = LSTMClassifier()
    print(model)
    dummy = torch.randint(0, VOCAB_SIZE, (8, 128))
    print("Output shape:", model(dummy).shape)
