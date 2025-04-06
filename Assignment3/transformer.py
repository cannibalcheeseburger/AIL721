
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt



seed = 43
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


batch_size = 64


def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', str(text).lower())


train_df = pd.read_csv("Datasets/TrainData.csv")
test_df = pd.read_csv("Datasets/TestLabels.csv")

train_df.dropna(subset=['Text', 'Category'], inplace=True)
test_df.dropna(subset=['Text', 'Label - (business, tech, politics, sport, entertainment)'], inplace=True)

print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

train_texts = train_df['Text'].tolist()
train_labels = train_df['Category'].tolist()
test_texts = test_df['Text'].tolist()
test_labels = test_df['Label - (business, tech, politics, sport, entertainment)'].tolist()


train_tokenized = [simple_tokenizer(t) for t in train_texts]
test_tokenized = [simple_tokenizer(t) for t in test_texts]


lengths = [len(inner_array) for inner_array in train_tokenized]
max_len = int(np.percentile(lengths, 90))


class NewsDataset(Dataset):
    def __init__(self, tokenized_text, labels, vocab=None, label2idx=None, max_len=300):
        self.texts = tokenized_text
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


train_data = NewsDataset(train_tokenized, train_labels)
test_data = NewsDataset(test_tokenized, test_labels, vocab=train_data.vocab, label2idx=train_data.label2idx)


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_classes=5,
                 num_layers=4, num_heads=8, max_len=300, pos_embed=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed_enabled = pos_embed
        
        # Embedding layers
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        if pos_embed:
            self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification layer
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        positions = torch.arange(0, x.size(1), dtype=torch.long).unsqueeze(0).to(device)
        
        x_embed = self.word_embedding(x)
        if self.pos_embed_enabled:
            x_embed += self.position_embedding(positions)
        
        x_transformed = self.transformer_encoder(x_embed)
        x_pooled = x_transformed.mean(dim=1)  # Mean pooling across sequence length
        
        return self.classifier(x_pooled)




def evaluate_model(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds ,all_labels = [] ,[]
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels,all_preds,average='micro')
    return avg_loss, accuracy,f1


def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    return avg_train_loss , train_accuracy


EPOCHS = 50
lr = 0.0003
weight_decay = 1e-3


model_params = {
    "vocab_size": len(train_data.vocab),
    "embed_dim": 512,
    "num_classes": len(train_data.label2idx),
    "num_layers": 1,
    "num_heads": 8,
    "max_len": max_len,
    "pos_embed": True
}

model = TextTransformer(**model_params).to(device)



criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0002)




patience = 15 
def analyze_encoder_blocks():
    """Analyze performance with different numbers of encoder blocks"""
    encoder_results = {}
    
    for num_layers in [2, 4, 6]:
        print(f"\n=== Training with {num_layers} encoder blocks ===")
        
        # Model configuration
        model_params = {
            "vocab_size": len(train_data.vocab),
            "embed_dim": 512,
            "num_classes": len(train_data.label2idx),
            "num_layers": num_layers,
            "num_heads": 8,
            "max_len": max_len,
            "pos_embed": True
        }
        
        # Initialize fresh model
        model = TextTransformer(**model_params).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0002)
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            avg_train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
            avg_val_loss, val_acc, val_f1 = evaluate_model(model, test_loader, criterion)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        encoder_results[num_layers] = best_f1
    
    return encoder_results

def analyze_positional_embeddings():
    """Analyze performance with/without positional embeddings"""
    pos_results = {}
    
    for use_pos in [True, False]:
        print(f"\n=== Training {'with' if use_pos else 'without'} positional embeddings ===")
        
        # Model configuration
        model_params = {
            "vocab_size": len(train_data.vocab),
            "embed_dim": 512,
            "num_classes": len(train_data.label2idx),
            "num_layers": 2,  # Default layer count
            "num_heads": 8,
            "max_len": max_len,
            "pos_embed": use_pos
        }
        
        # Initialize fresh model
        model = TextTransformer(**model_params).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0002)
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            avg_train_loss, train_acc = train_model(model, train_loader, criterion, optimizer)
            avg_val_loss, val_acc, val_f1 = evaluate_model(model, test_loader, criterion)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        pos_results['With Pos' if use_pos else 'Without Pos'] = best_f1
    
    return pos_results

# Add this visualization function
def plot_results(results, title, xlabel):
    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values())
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Micro F1 Score")
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Run analyses after your single model training
print("\n" + "="*50)
print("Analyzing Encoder Blocks...")
encoder_results = analyze_encoder_blocks()
print("\nEncoder Block Results:")
for layers, f1 in encoder_results.items():
    print(f"{layers} layers: F1 = {f1:.4f}")

print("\n" + "="*50)
print("Analyzing Positional Embeddings...")
pos_results = analyze_positional_embeddings()
print("\nPositional Embedding Results:")
for config, f1 in pos_results.items():
    print(f"{config}: F1 = {f1:.4f}")

# Plot results
plot_results(encoder_results, "Effect of Encoder Blocks", "Number of Encoder Layers")
plot_results(pos_results, "Effect of Positional Embeddings", "Configuration")



patience = 15 
best_val_loss = float('inf')
counter = 0  
for epoch in range(EPOCHS):
    avg_train_loss , train_accuracy = train_model(model, train_loader, criterion, optimizer)
    avg_val_loss, val_accuracy ,f1 = evaluate_model(model, test_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | f1: {f1:.2f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0 
        best_model = model.state_dict()  
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
        break


model.load_state_dict(best_model)


# Final evaluation and confusion matrix
avg_val_loss, val_accuracy, f1 = evaluate_model(model, test_loader, criterion)
print(f"\nFinal Micro F1 Score = {f1:.4f}")
label_names = [label for label, _ in sorted(train_data.label2idx.items(), key=lambda x: x[1])]









