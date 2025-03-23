import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import csv
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt', type=str, required=True)
parser.add_argument('--test_imgs', type=str, required=True)
args = parser.parse_args()

torch.manual_seed(420)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 128

test_path = args.test_imgs 
output_path = 'seg_maps'
csv_path = 'submission.csv'

with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

n=2
r=100
DROPOUT = 0.23279759530518396

os.makedirs(output_path, exist_ok=True)

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ImageDataset(root_dir=test_path, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

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
        
        self.layer1 = self.add_layer(32, n, stride=1)
        self.layer2 = self.add_layer(64, n, stride=2)
        self.layer3 = self.add_layer(128, n, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, r)
        self.dropout = nn.Dropout(p=DROPOUT)

    def add_layer(self, out_channels, num_blocks, stride):
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
model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
model.eval()


def grad_cam(model, input_image):
    x = input_image.unsqueeze(0).to(device)
    
    with torch.enable_grad():
        x.requires_grad_()
        features = model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(x))))))
        output = model.fc(model.avg_pool(features).view(x.size(0), -1))
        
        model.zero_grad()
        score = torch.max(output)
        score.backward()
        
        gradients = model.layer3[-1].conv2.weight.grad.data
    
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    
    return heatmap.detach().cpu().numpy()


def create_binary_mask(heatmap, percentile=43):
    threshold = np.percentile(heatmap, percentile)
    return (heatmap > threshold).astype(np.uint8) * 255



all_predictions = []

for images, img_names in test_dataloader:
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    for image, img_name, pred in zip(images, img_names, predicted):
        heatmap = grad_cam(model, image)
        binary_mask = create_binary_mask(heatmap)
        binary_mask = cv2.resize(binary_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        output_mask_path = os.path.join(output_path, f"{img_name}")
        cv2.imwrite(output_mask_path, binary_mask)
        
        predicted_label = class_labels[str(pred.item())]
        all_predictions.append((img_name, predicted_label))

sorted_predictions = sorted(all_predictions, key=lambda x: x[0])

with open(csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['image_name', 'label'])
    csvwriter.writerows(sorted_predictions)

print("Processing complete. Segmentation maps saved in 'seg_maps' folder and predictions saved in 'submission.csv'.")