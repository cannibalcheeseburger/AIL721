import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------- Tokenizer -------------
def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', str(text).lower())

# ------------- Dataset Class -------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, label2idx=None, max_len=300):
        self.texts = [simple_tokenizer(t) for t in texts]
        self.max_len = max_len

        if vocab is None:
            words = [word for text in self.texts for word in text]
            word_freq = Counter(words)
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            for word in word_freq:
                self.vocab[word] = len(self.vocab)
        else:
            self.vocab = vocab

        self.texts = [self.encode(text) for text in self.texts]

        if label2idx is None:
            unique_labels = sorted(set(label for label in labels if pd.notna(label)))
            self.label2idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label2idx = label2idx

        self.labels = [self.label2idx[label] for label in labels if pd.notna(label)]

    def encode(self, tokens):
        encoded = [self.vocab.get(tok, self.vocab['<UNK>']) for tok in tokens]
        return encoded[:self.max_len] + [0]*(self.max_len - len(encoded))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

# ------------- Model -------------
class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(256, 1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)              # (B, T, D)
        x = x.permute(0, 2, 1)             # (B, D, T)
        x = self.relu(self.conv(x))       # (B, C, T)
        x = x.permute(0, 2, 1)             # (B, T, C)
        lstm_out, _ = self.lstm(x)        # (B, T, 2H)
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)          # (B, 2H)
        return self.fc(context)           # (B, num_classes)

# ------------- Train Function -------------
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# ------------- Evaluate Function -------------
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in loader:
            texts = texts.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    f1 = f1_score(all_labels, all_preds, average='micro')
    cm = confusion_matrix(all_labels, all_preds)
    return f1, cm, all_labels, all_preds

# ------------- Confusion Matrix Plot -------------
def plot_confusion_matrix(cm, label_names, save_path="outputs/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Confusion matrix saved to: {save_path}")

# ------------- Main -------------
if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv("Datasets/TrainData.csv")
    test_df = pd.read_csv("Datasets/TestLabels.csv")
    label_col = test_df.columns[-1]

    # Drop rows with missing data
    train_df.dropna(subset=['Text', 'Category'], inplace=True)
    test_df.dropna(subset=['Text', label_col], inplace=True)

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    train_texts = train_df['Text'].tolist()
    train_labels = train_df['Category'].tolist()
    test_texts = test_df['Text'].tolist()
    test_labels = test_df[label_col].tolist()

    # Dataset and Dataloader
    train_data = NewsDataset(train_texts, train_labels)
    test_data = NewsDataset(test_texts, test_labels, vocab=train_data.vocab, label2idx=train_data.label2idx)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_Classifier(len(train_data.vocab), embed_dim=100, num_classes=len(train_data.label2idx)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        train_model(model, train_loader, criterion, optimizer, device)
        f1, _, _, _ = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}: Micro F1 Score = {f1:.4f}")

    # Final evaluation and confusion matrix
    f1, cm, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"\nFinal Micro F1 Score = {f1:.4f}")
    label_names = [label for label, _ in sorted(train_data.label2idx.items(), key=lambda x: x[1])]
    plot_confusion_matrix(cm, label_names)
