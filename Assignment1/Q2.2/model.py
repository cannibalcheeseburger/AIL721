import torch.nn as nn

class WineModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(12,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,11)
        )
    
    def forward(self,X):
        return self.model(X)