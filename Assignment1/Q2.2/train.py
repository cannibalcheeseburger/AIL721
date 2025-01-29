import pandas as pd
from dataset import WineDataset
from model import WineModel
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import Counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    args = parser.parse_args()

    model = WineModel()
    n_epochs = min(int(args.n_epochs), 200)

    lr = 0.01
    train_csv_path = "./Train.csv"

    csv_data = pd.read_csv(train_csv_path)
    mid = round(len(csv_data)*0.2)
    train_dataset  = WineDataset(csv_data)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)

    for epoch in range(n_epochs):
        train_loss = 0
        for batch_data , batch_labels in train_loader:
            outputs = model(batch_data)
            loss = loss_function(outputs,batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

        print(f'Epoch: {epoch + 1} , Loss: {train_loss/len(train_loader)}')


    model_path = "./model_weights.pth"
    torch.save(model.state_dict(), model_path)