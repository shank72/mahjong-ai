import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A single Residual Layer Block for processing dense vectors.
    Keeps dimensions identical from input to output to allow the skip addition.
    """
    def __init__(self, dim, dropout_rate=0.3):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x  # Store the raw incoming tensor as our identity shortcut
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        out += identity  # Add the uncorrupted input back into the processed features
        out = self.relu(out)
        return out


# =========================================================================
# POLICY 1: The Discard Engine (ResNet Backbone)
# =========================================================================
class MahjongDiscardResNet(nn.Module):
    """
    Output: 34 values (Raw logits for CrossEntropyLoss)
    Uses a deep hidden space and multiple blocks to evaluate 34 discrete discard vectors.
    """
    def __init__(self, input_dim=240, hidden_dim=512, num_blocks=3, num_classes=34, dropout_rate=0.2):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim=hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)


# =========================================================================
# POLICY 2: The Decision Engine (Used for Chi, Pon, and all Kans)
# =========================================================================
class MahjongBinaryResNet(nn.Module):
    """
    Output: 2 values ([0: Skip/No, 1: Call/Yes])
    Lightweight, highly optimized tactical calling evaluation network.
    """
    def __init__(self, input_dim=240, hidden_dim=128, num_blocks=2, num_classes=2, dropout_rate=0.5):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim=hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)

    
# =========================================================================
# POLICY 3: The Decision Engine (Riichi)
# =========================================================================
class MahjongRiichiResNet(nn.Module):
    """
    Output: 2 values ([0: Skip/No, 1: Call/Yes])
    Mid-tier capacity architecture balancing raw tile patterns against strategic value.
    """
    def __init__(self, input_dim=240, hidden_dim=256, num_blocks=2, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dim=hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)