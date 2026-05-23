import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Absolute imports from your specific model folder
from nn_model import MahjongDiscardMLP, MahjongBinaryMLP

# Argument Parsing for dynamic table orchestration
parser = argparse.ArgumentParser(description="Unified Mahjong Engine Training Script")
parser.add_argument("--task", type=str, default="discard", choices=["discard", "pon", "chi", "riichi", "daiminkan", "shouminkan", "ankan"],
                    help="Target strategy table to optimize")
parser.add_argument("--data_dir", type=str, default="datasets/full-context", help="Path to pre-built numpy tensors")
args = parser.parse_args()

TASK = args.task.lower()
IS_BINARY = TASK != "discard"

# HYPERPARAMETERS
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
DROPOUT_RATE = 0.2
PATIENCE = 10
CHECKPOINT_DIR = f"checkpoints/{TASK}"
LOG_DIR = f"logs/full-context/{TASK}"
# ------------------------------------------------

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def calculate_top_k_accuracy(logits, targets, k=3):
    """Computes the precision@k for the specified values of k"""
    if IS_BINARY and k >= 2:
        return 0.0 # Return empty fallback for binary sets
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()

# Load specific array pairings dynamically
print(f"[INFO] Loading target datasets for: {TASK.upper()}")
X_path = os.path.join(args.data_dir, f"{TASK}_X.npy")
y_path = os.path.join(args.data_dir, f"{TASK}_y.npy")

X = np.load(X_path)
y = np.load(y_path)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# Initialize System Matching the Required Task Topology
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

if IS_BINARY:
    model = MahjongBinaryMLP(dropout_rate=DROPOUT_RATE).to(device)
    print("[MODEL] Instantiated MahjongBinaryMLP (2 Output Target Classes)")
else:
    model = MahjongDiscardMLP(dropout_rate=DROPOUT_RATE).to(device)
    print("[MODEL] Instantiated MahjongDiscardMLP (34 Output Target Tile Classes)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

best_val_loss = float('inf')
patience_counter = 0

print(f"[INFO] Starting training pipeline for execution framework: {TASK}")
log_file = open(os.path.join(LOG_DIR, "training_log.csv"), "w")
if IS_BINARY:
    log_file.write("epoch,train_loss,val_loss,val_acc_top1\n")
else:
    log_file.write("epoch,train_loss,val_loss,val_acc_top1,val_acc_top3\n")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        logits = model(xb)
        loss = criterion(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
    # Validation phase
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
        val_logits = model(X_val_dev)
        val_loss = criterion(val_logits, y_val_dev)
        total_val_loss = val_loss.item()
        
        val_acc_top1 = calculate_top_k_accuracy(val_logits, y_val_dev, k=1)
        val_acc_top3 = calculate_top_k_accuracy(val_logits, y_val_dev, k=3)

    if IS_BINARY:
        print(f"Epoch {epoch+1:02d} | Train Loss: {total_train_loss:.2f} | Val Loss: {total_val_loss:.2f} | Val Acc: {val_acc_top1 * 100:.2f}%")
        log_file.write(f"{epoch+1},{total_train_loss},{total_val_loss},{val_acc_top1}\n")
    else:
        print(f"Epoch {epoch+1:02d} | Train Loss: {total_train_loss:.2f} | Val Loss: {total_val_loss:.2f} | Val Top-1 Acc: {val_acc_top1:.4f} | Val Top-3 Acc: {val_acc_top3:.4f}")
        log_file.write(f"{epoch+1},{total_train_loss},{total_val_loss},{val_acc_top1},{val_acc_top3}\n")
    
    log_file.flush()

    # Checkpoint logic
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        patience_counter = 0
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
        }, checkpoint_path)
    else:
        patience_counter += 1
        
    if patience_counter >= PATIENCE:
        print(f"[INFO] Early stopping triggered at epoch {epoch+1}.")
        break

log_file.close()
print("[DONE] Training complete.")

# =====================================================================
# FINAL EVALUATION: TRAIN VS TEST COMPARISON

print("\n" + "="*50)
print(f"[INFO] Running Final Evaluation (Train vs Test) for {TASK.upper()}...")
print("="*50)

checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
if os.path.exists(checkpoint_path):
    print(f"[INFO] Loading best weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print("[WARNING] No checkpoint found! Evaluating final epoch instead.")

model.eval()

# Evaluate on full Training Set
train_eval_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
total_train_top1 = 0
total_train_top3 = 0

with torch.no_grad():
    for xb, yb in train_eval_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_train_top1 += calculate_top_k_accuracy(logits, yb, k=1) * xb.size(0)
        total_train_top3 += calculate_top_k_accuracy(logits, yb, k=3) * xb.size(0)

final_train_acc1 = total_train_top1 / len(X_train)
final_train_acc3 = total_train_top3 / len(X_train)

# Evaluate on full Validation Set
val_eval_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
total_val_top1 = 0
total_val_top3 = 0

with torch.no_grad():
    for xb, yb in val_eval_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_val_top1 += calculate_top_k_accuracy(logits, yb, k=1) * xb.size(0)
        total_val_top3 += calculate_top_k_accuracy(logits, yb, k=3) * xb.size(0)

final_val_acc1 = total_val_top1 / len(X_val)
final_val_acc3 = total_val_top3 / len(X_val)

# Final comparative terminal display output
print("\n### METRIC SUMMARY ###")
print(f"Train Dataset Size: {len(X_train)} | Val Dataset Size: {len(X_val)}")
print("-" * 50)
print(f"Top-1 Accuracy / Decision Precision Match:")
print(f"  -> Train Accuracy: {final_train_acc1 * 100:.2f}%")
print(f"  -> Val/Test Accuracy: {final_val_acc1 * 100:.2f}%")
print(f"  -> Generalization Gap: {(final_train_acc1 - final_val_acc1) * 100:.2f}%")

if not IS_BINARY:
    print("-" * 50)
    print(f"Top-3 Accuracy (Viable Options):")
    print(f"  -> Train Top-3 Accuracy: {final_train_acc3 * 100:.2f}%")
    print(f"  -> Val/Test Top-3 Accuracy: {final_val_acc3 * 100:.2f}%")
print("="*50)

'''
python models/full-context/train_nn.py --task discard

python models/full-context/train_nn.py --task chi
python models/full-context/train_nn.py --task pon

python models/full-context/train_nn.py --task riichi

python models/full-context/train_nn.py --task daiminkan
python models/full-context/train_nn.py --task ankan
python models/full-context/train_nn.py --task shouminkan

'''