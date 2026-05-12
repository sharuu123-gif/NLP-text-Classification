import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
import os

print('='*50)
print('DistilBERT - Fast Text Classification')
print('='*50)

# Very small sample for speed
print('Loading data...')
train_df = pd.read_csv('data/processed/train_clean.csv').dropna(subset=['clean_text']).sample(500, random_state=42)
test_df  = pd.read_csv('data/processed/test_clean.csv').dropna(subset=['clean_text']).sample(100, random_state=42)
print(f'Train: {len(train_df)}  Test: {len(test_df)}')

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# DistilBERT is 40% faster and smaller than BERT
print('Loading DistilBERT tokenizer...')
tokenizer    = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_enc    = tokenizer(list(train_df['clean_text']), padding=True, truncation=True, max_length=32, return_tensors='pt')
test_enc     = tokenizer(list(test_df['clean_text']),  padding=True, truncation=True, max_length=32, return_tensors='pt')
train_ds     = TextDataset(train_enc, train_df['label'].tolist())
test_ds      = TextDataset(test_enc,  test_df['label'].tolist())
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)
print('✅ Tokenization done!')

print('Loading DistilBERT model...')
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
model     = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

print()
print('Training DistilBERT...')
for epoch in range(2):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    print(f'Epoch {epoch+1}/2 | Loss: {total_loss/len(train_loader):.4f}')

print()
print('Evaluating...')
model.eval()
correct = total = 0
with torch.no_grad():
    for batch in test_loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop('labels')
        preds  = model(**batch).logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
acc = correct / total

os.makedirs('saved_models/bert_model', exist_ok=True)
model.save_pretrained('saved_models/bert_model')
tokenizer.save_pretrained('saved_models/bert_model')

print()
print('='*50)
print('DISTILBERT COMPLETE!')
print(f'Accuracy : {acc:.2%}')
print('Saved    -> saved_models/bert_model/')
print('='*50)