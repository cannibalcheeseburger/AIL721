import time
start_time = time.time()
import argparse
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import os
import json
import torch.nn as nn
from torch.utils.data import random_split, DataLoader



parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str, required=True)
parser.add_argument('--model_ckpt', type=str, required=True)
parser.add_argument('--out', type=str, required=True)


args = parser.parse_args()

sys.stdout = open(f'./outputs/output_{args.out}.txt', 'w', buffering=1)
sys.stderr = sys.stdout

torch.manual_seed(420)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_folder = args.train_data #'Butterfly/train'
test_folder = 'Butterfly/test'

n = 2
r = 100
learning_rate = 0.3938867103103102
DROPOUT = 0.23279759530518396

EPOCHS = 55
BATCH_SIZE = 128


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
class_to_idx = train_dataset.class_to_idx
train_size = int(0.8 * len(train_dataset))  
val_size = len(train_dataset) - train_size 
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
test_dataset = datasets.ImageFolder(root=test_folder, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

idx_to_class = {v: k for k, v in class_to_idx.items()}
with open('class_labels.json', 'w') as f:
    json.dump(idx_to_class, f)


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
        self.dropout = nn.Dropout(p=DROPOUT)

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


model = ResNet(n, r).to(device)

criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.8497635194529174, weight_decay=0.0005431283741227301)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4053237616926611, patience=5)

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

print('Starting Training')
for epoch in range(EPOCHS):
    print(f'Epoch: {epoch+1}/{EPOCHS}')
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion)
    print(f'Train_acc: {train_acc:.2f}%, Train_loss: {train_loss:.4f}')

    val_acc, val_loss = evaluate_model(model, val_loader, criterion)
    print(f'Val_acc: {val_acc:.2f}%, Val_loss: {val_loss:.4f}')

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Current Learning Rate: {current_lr:.6f}')


torch.save(model.state_dict(), os.path.join(args.model_ckpt,'resnet_model.pth'))
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time/60:.4f} minutes")

test_acc, test_loss = evaluate_model(model, test_loader, criterion)
print(f'Test_acc: {test_acc:.2f}%, Test_loss: {test_loss:.4f}')
