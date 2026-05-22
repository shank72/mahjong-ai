import torch
import torch.nn as nn

class MahjongMLP(nn.Module):
    def __init__(self, input_dim=187, num_classes=34, dropout_rate=0.2): # drop oroginally 0.3
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    

# 5000 samples
# 10 epochs:  .18199 accuracy
# 30 epochs:  .23299 accuracy
# 50 epochs:  .23600 accuracy
# 75 epochs:  .24799 accuracy
# 100 epochs: .25100 accuracy

# 50000 samples
# 100 epochs:  .39809 accuracy