import torch
import torch.nn as nn

class MahjongDiscardMLP(nn.Module):
    """
    POLICY 1: The Discard Engine
    Output: 34 values (Raw logits for CrossEntropyLoss)
    """
    def __init__(self, input_dim=187, num_classes=34, dropout_rate=0.2):
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


class MahjongBinaryMLP(nn.Module):
    """
    POLICY 2: The Decision Engine (Used for Chi, Pon, Riichi, and all Kans)
    Output: 2 values ([0: Skip/No, 1: Call/Yes])
    """
    def __init__(self, input_dim=187, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), 
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
    
class MahjongRichiMLP(nn.Module):
    """
    POLICY 2: The Decision Engine (Used for Chi, Pon, Riichi, and all Kans)
    Output: 2 values ([0: Skip/No, 1: Call/Yes])
    """
    def __init__(self, input_dim=187, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Dropout(dropout_rate),

            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)