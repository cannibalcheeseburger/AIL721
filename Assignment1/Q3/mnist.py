import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset,random_split
import pickle
import torch.optim as optim


#Loading the training data
with open('train.pkl', 'rb') as file:
    train_data = pickle.load(file)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



# Step 1:Dataset Creation
##############################################
class MNISTDataset(Dataset):
    def __init__(self, data,):
        self.data = data

    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]        
        return image, label
    
#Implement your logic here
##############################################

train_dataset = MNISTDataset(train_data)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

# Use random_split to divide the dataset
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=64,shuffle=False)

# Step 2: MLP creation
##############################################
class MLP(nn.Module):
    def __init__(self,input_size=28*28,hidden_units=64,output_size=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,output_size),
        )
    
    def forward(self,x):
        x = x.view(x.size(0), -1)  
        return self.model(x)
    
#Implement your logic here 
# (make sure the model returns logits only not the prediction)
##############################################

# Not to be changed
epochs = 50


# Step 3: Loading the data to Dataloader and hyperparammeters selection
##############################################
input_size=28*28
hidden_units=64
output_size=10
model = MLP(input_size,hidden_units,output_size)


#Implement your logic here
##############################################
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_function = nn.CrossEntropyLoss()

# Step 4: Training the model and saving it.
##############################################
for epoch in range(epochs):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    val_loss = 0.0
    total_val = 0
    correct_val =0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    with torch.no_grad():
        for batch_data, batch_labels in val_dataloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss  = loss_function(outputs,batch_labels)
            val_loss+=loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += batch_labels.size(0)
            correct_val += (predicted == batch_labels).sum().item()

    
    train_loss = running_loss / len(train_dataloader)
    val_loss = val_loss / len(val_dataloader)
    train_accuracy = 100 * correct / total
    val_accuracy = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}% \n")

model_path = "./model_weights.pth"
torch.save(model.state_dict(), model_path)

#Implement your logic here
##############################################


#Inference (Don't change the code)
def evaluate(model,test_data_path):
    with open(test_data_path, 'rb') as file:
        test_data = pickle.load(file)
    test_dataset = MNISTDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) 
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100*correct/total 
    return accuracy

accuracy = evaluate(model,'test.pkl')
print(f"accuracy: {accuracy}%")
