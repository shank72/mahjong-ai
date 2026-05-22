import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Core architectural imports
from nn_model import MahjongDiscardMLP, MahjongBinaryMLP

DATA_DIR = "datasets/full-context"
CHECKPOINT_ROOT = "checkpoints"
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All tasks to evaluate systematically
TASKS = ["discard", "pon", "chi", "riichi", "daiminkan", "shouminkan", "ankan"]

def calculate_top_k(logits, targets, k=1, is_binary=False):
    if is_binary and k > 1:
        return 0.0
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()

print("=" * 60)
print(f"STARTING GLOBAL MAHJONG ENGINE EVALUATION (Device: {DEVICE})")
print("=" * 60)

print(f"{'TASK':<12} | {'VAL SIZE':<8} | {'TOP-1 ACC':<11} | {'TOP-3 ACC':<11}")
print("-" * 60)

for task in TASKS:
    x_path = os.path.join(DATA_DIR, f"{task}_X.npy")
    y_path = os.path.join(DATA_DIR, f"{task}_y.npy")
    ckpt_path = os.path.join(CHECKPOINT_ROOT, task, "best_model.pt")
    
    # 1. Verification step: Ensure files exist before attempting load
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print(f"{task:<12} | Missing dataset files (.npy). Skipping.")
        continue
    if not os.path.exists(ckpt_path):
        print(f"{task:<12} | Missing model checkpoint. Skipping.")
        continue
        
    is_binary = task != "discard"
    
    # 2. Data Preparation Pipeline
    X = np.load(x_path)
    y = np.load(y_path)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Isolate the exact same test split used during training
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Model Architecture Matching Blueprint
    if is_binary:
        model = MahjongBinaryMLP(dropout_rate=0.0).to(DEVICE)
    else:
        model = MahjongDiscardMLP(dropout_rate=0.0).to(DEVICE)
        
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Batched Validation Evaluation
    total_top1, total_top3, sample_count = 0.0, 0.0, 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            
            bs = xb.size(0)
            total_top1 += calculate_top_k(logits, yb, k=1, is_binary=is_binary) * bs
            if not is_binary:
                total_top3 += calculate_top_k(logits, yb, k=3, is_binary=is_binary) * bs
            sample_count += bs

    final_acc1 = (total_top1 / sample_count) * 100
    final_acc3 = (total_top3 / sample_count) * 100
    
    # 5. Render row results
    top3_str = f"{final_acc3:.2f}%" if not is_binary else "N/A (Binary)"
    print(f"{task:<12} | {sample_count:<8} | {final_acc1:.2f}%{'':<4} | {top3_str}")

print("=" * 60)
print("[INFO] Evaluation sequence finished.")