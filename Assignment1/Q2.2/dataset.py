import torch
from torch.utils.data import Dataset

class WineDataset(Dataset):
    def __init__(self, csv_data):
        X = csv_data.copy()
        y = torch.from_numpy(X['quality'].values)
        unique_categories = list(X['variety'].unique())  
        category_to_index = {var: idx for idx, var in enumerate(unique_categories)}
        X['variety'] = X['variety'].map(category_to_index)
        X = torch.from_numpy(X.drop(labels='quality',axis = 1).values)
        self.data = X.float()
        self.labels = y.long()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]