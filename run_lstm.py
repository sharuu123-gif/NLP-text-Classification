import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import os

print('Loading data...')
train_df = pd.read_csv('data/processed/train_clean.csv').dropna(subset=['clean_text']).sample(500, random_state=42)
test_df  = pd.read_csv('data/processed/test_clean.csv').dropna(subset=['clean_text']).sample(100, random_state=42)

all_words = ' '.join(train_df['clean_text']).split()
vocab     = ['<PAD>','<OOV>'] + [w for w,c in Counter(all_words).most_common(5000)]
word2idx  = {w:i for i,w in enumerate(vocab)}

def encode(text, max_len=20):
    tokens = str(text).split()[:max_len]
    ids    = [word2idx.get(t, 1) for t in tokens]
    ids   += [0] * (max_len - len(ids))
    return ids

X_train = torch.tensor([encode(t) for t in train_df['clean_text']], dtype=torch.long)
X_test  = torch.tensor([encode(t) for t in test_df['clean_text']],  dtype=torch.long)
y_train = torch.tensor(train_df['label'].values, dtype=torch.long)
y_test  = torch.tensor(test_df['label'].values,  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=500, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=100)

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 16, padding_idx=0)
        self.lstm      = nn.LSTM(16, 32, batch_first=True, bidirectional=True)
        self.fc        = nn.Linear(64, 4)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

model     = BiLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

print('Training...')
for epoch in range(3):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/3 | Loss: {total_loss:.4f}')

model.eval()
correct = total = 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds   = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total   += yb.size(0)
acc = correct / total

os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/lstm_model.pth')
print('='*40)
print('BILSTM COMPLETE!')
print(f'Accuracy : {acc:.2%}')
print('='*40)