import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import matplotlib.pyplot as plt

# Loading data
file_path = "train.dat"
columns = [
    "frequency", 
    "attack_angle", 
    "chord_length", 
    "free_stream_velocity", 
    "suction_side_displacement_thickness", 
    "scaled_sound_pressure"
]
data = pd.read_csv(file_path, sep="\t", header=None, names=columns)


# Step: 1 Features and target values
##############################################
X = data[["frequency", "attack_angle", "chord_length", "free_stream_velocity", "suction_side_displacement_thickness"]]
y = data["scaled_sound_pressure"]
##############################################


# Step: 2 Training-Validation split
##############################################
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
##############################################



# Step: 3 Normalizing features
##############################################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
##############################################


# Converting to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

# Step: 4 Defining the NN
##############################################
class LinearRegressionModel(nn.Module):
    #change this
    def __init__(self,num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features,10),
            nn.ReLU(),
            nn.Linear(10,1),
            nn.ReLU()
        )
    
    def forward(self,X):
        return self.model(X)
##############################################
    

# Step: 5 Hyperparameters (learning_rate, batch_size, Loss_function, optimizer)
##############################################
model = LinearRegressionModel(X_train.shape[1])
lr = 0.1
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
batch_size = 32

##############################################


# Not to be changed
epochs = 300

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


# Step: 6 Training loop
################################################################
train_loss = []
val_loss = []
for i in range(epochs): # Total training epoch is 10000
    
    outputs = model(X_train)

    loss = loss_function(outputs.squeeze(),y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    print(f'Epoch: {i + 1} ,Train Loss: {loss.item()}')
    train_loss.append(loss.item()) # Log the training loss for the epoch

    with torch.no_grad():
        val_out = model(X_val)
        val_l = loss_function(val_out.squeeze(),y_val)
        print(f'Val Loss:{val_l}')
        val_loss.append(val_l)

################################################################


# Step: 7 Train and Val Loss plot
##############################################
plt.xlabel("Epoch")
plt.ylabel('Loss')
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['Train Loss','Validation Loss'])
plt.savefig(f"./loss.png")
##############################################


# Step: 8 Save the Model (Format: EntryNumber_model.pkl )
##############################################
torch.save(model,'./2024AIB2289_model.pkl')
##############################################








