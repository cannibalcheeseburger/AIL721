import sys
sys.stdout = open(f'./tune.txt', 'w', buffering=1)
sys.stderr = sys.stdout


import optuna
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam,SGD
from torch.nn import CrossEntropyLoss
import random
import json
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(420)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_folder = 'Butterfly/train'
val_folder = 'Butterfly/valid'
test_folder = 'Butterfly/test'

n = 2
r = 100
EPOCHS = 55
BATCH_SIZE = 128


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, n, r):
        super(ResNet, self).__init__()
        self.in_channels = 32
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self.make_layer(32, n, stride=1)
        self.layer2 = self.make_layer(64, n, stride=2)
        self.layer3 = self.make_layer(128, n, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, r)
        self.dropout = nn.Dropout()

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_folder, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_folder, transform=test_transforms)
test_dataset = datasets.ImageFolder(root=test_folder, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss

def objective(trial):
    # Define the hyperparameters to be tuned
    lr = trial.suggest_uniform('lr', 0.01, 0.7)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    patience = trial.suggest_int('patience', 3, 7)
    factor = trial.suggest_uniform('factor', 0.1, 0.5)

    # Update the model and optimizer with the new hyperparameters
    model = ResNet(n, r).to(device)
    model.dropout.p = dropout
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)

    for epoch in range(EPOCHS):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_loss = evaluate_model(model, val_loader, criterion)
        
        scheduler.step(val_acc)
        
    test_acc, _ = evaluate_model(model, test_loader, criterion)
    
    return test_acc

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

# Print the best hyperparameters and result
print('Best trial:')
trial = study.best_trial
print('Test Accuracy: ', trial.value)
print('Best hyperparameters:')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Train the final model with the best hyperparameters
best_lr = trial.params['lr']
best_dropout = trial.params['dropout']
best_momentum = trial.params['momentum']
best_weight_decay = trial.params['weight_decay']
best_patience = trial.params['patience']
best_factor = trial.params['factor']

model = ResNet(n, r).to(device)
model.dropout.p = best_dropout
criterion = CrossEntropyLoss()

optimizer = SGD(model.parameters(), lr=best_lr, momentum=best_momentum, weight_decay=best_weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=best_factor, patience=best_patience, verbose=True)

# Train the model with the best hyperparameters
best_val_acc = 0
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}/{EPOCHS}')
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion)
    print(f'Train_acc: {train_acc:.2f}%, Train_loss: {train_loss:.4f}')

    val_acc, val_loss = evaluate_model(model, val_loader, criterion)
    print(f'Val_acc: {val_acc:.2f}%, Val_loss: {val_loss:.4f}')

    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'final_best_model.pth')

# Load the best model and evaluate on test set
model.load_state_dict(torch.load('final_best_model.pth'))
test_acc, test_loss = evaluate_model(model, test_loader, criterion)
print(f'Final Test_acc: {test_acc:.2f}%, Test_loss: {test_loss:.4f}')
