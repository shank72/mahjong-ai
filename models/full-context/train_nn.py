import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# NEW: Added structural metrics primitives for multi-dimensional tracking
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Absolute imports from your specific model folder
from nn_model import MahjongDiscardResNet, MahjongBinaryResNet, MahjongRiichiResNet

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
EPOCHS = 100
DISCARD_DROPOUT_RATE = 0.2
PATIENCE = 10

# Task-specific regularization mappings
if TASK == "riichi":
    BINARY_DROPOUT_RATE = 0.5  # Force aggressive weight feature mutation
    WEIGHT_DECAY = 1e-3        # Tighten L2 constraints to crush memorization
    
elif TASK in ["chi", "pon"]:
    BINARY_DROPOUT_RATE = 0.5
    WEIGHT_DECAY = 5e-4
    
else:  # Keep highly successful settings for Discard and Kans
    BINARY_DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = f"checkpoints/full-context/{TASK}"
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

# NEW: Unified extraction wrapper for strategic sub-metrics
def calculate_macro_metrics(logits, targets):
    """Computes Macro Precision, Recall, and F1-score safely across active target pools"""
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels = targets.cpu().numpy()
    
    # zero_division=0 keeps script alive if a class is completely untouched during an epoch
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, preds, average='macro', zero_division=0
    )
    return precision, recall, f1

# Load specific array pairings dynamically
print(f"[INFO] Loading target datasets for: {TASK.upper()}")
X_path = os.path.join(args.data_dir, f"{TASK}_X.npy")
y_path = os.path.join(args.data_dir, f"{TASK}_y.npy")

X = np.load(X_path)
y = np.load(y_path)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# Initialize System Matching the Required Task Topology
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

if TASK == "riichi":
    model = MahjongRiichiResNet(dropout_rate=BINARY_DROPOUT_RATE).to(device)
    print("[MODEL] Instantiated MahjongRiichiResNet (2 Output Target Classes with Riichi-Specific Architecture)")
elif IS_BINARY:
    model = MahjongBinaryResNet(dropout_rate=BINARY_DROPOUT_RATE).to(device)
    print("[MODEL] Instantiated MahjongBinaryResNet (2 Output Target Classes)")
else:
    model = MahjongDiscardResNet(dropout_rate=DISCARD_DROPOUT_RATE).to(device)
    print("[MODEL] Instantiated MahjongDiscardResNet (34 Output Target Tile Classes)")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# CHANGED: Monitored target switched from raw validation loss to Macro F1 
# to aggressively combat class-imbalance blindspots.
best_val_f1 = -1.0 
patience_counter = 0

print(f"[INFO] Starting training pipeline for execution framework: {TASK}")
log_file = open(os.path.join(LOG_DIR, "training_log.csv"), "w")

# CHANGED: Added F1 columns to CSV header footprints
if IS_BINARY:
    log_file.write("epoch,train_loss,val_loss,val_acc_top1,val_macro_precision,val_macro_recall,val_macro_f1\n")
else:
    log_file.write("epoch,train_loss,val_loss,val_acc_top1,val_acc_top3,val_macro_precision,val_macro_recall,val_macro_f1\n")

start_time = time.time()

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
        
        total_train_loss += loss.item() * xb.size(0) 
        
    avg_train_loss = total_train_loss / len(X_train)
        
    # Validation phase
    model.eval()
    
    with torch.no_grad():
        X_val_dev, y_val_dev = X_val.to(device), y_val.to(device)
        val_logits = model(X_val_dev)
        val_loss = criterion(val_logits, y_val_dev)
        
        avg_val_loss = val_loss.item() 
        
        val_acc_top1 = calculate_top_k_accuracy(val_logits, y_val_dev, k=1)
        val_acc_top3 = calculate_top_k_accuracy(val_logits, y_val_dev, k=3)
        
        # NEW: Process deep precision footprints on validation step
        val_prec, val_rec, val_f1 = calculate_macro_metrics(val_logits, y_val_dev)

    # CHANGED: Enriched logging strings with detailed performance parameters
    if IS_BINARY:
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc_top1 * 100:.2f}% | Macro F1: {val_f1 * 100:.2f}%")
        log_file.write(f"{epoch+1},{avg_train_loss},{avg_val_loss},{val_acc_top1},{val_prec},{val_rec},{val_f1}\n")
    else:
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Top-1: {val_acc_top1*100:.2f}% | Top-3: {val_acc_top3*100:.2f}% | Macro F1: {val_f1*100:.2f}%")
        log_file.write(f"{epoch+1},{avg_train_loss},{avg_val_loss},{val_acc_top1},{val_acc_top3},{val_prec},{val_rec},{val_f1}\n")
    
    log_file.flush()

    # CHANGED: Early stopping now targets MAXIMIZING Validation Macro F1 instead of minimizing loss
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_macro_f1': best_val_f1,
        }, checkpoint_path)
    else:
        patience_counter += 1
        
    if patience_counter >= PATIENCE:
        print(f"[INFO] Early stopping triggered at epoch {epoch+1} based on validation F1 flatline.")
        break

log_file.close()
print("[DONE] Training complete.")

end_time = time.time()
total_duration = end_time - start_time

hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = total_duration % 60

# =====================================================================
# FINAL EVALUATION: ENRICHED REPORT OUTPUT

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

# Complete Training predictions capture block
train_eval_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
all_train_preds = []
all_train_trues = []

with torch.no_grad():
    for xb, yb in train_eval_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        all_train_preds.append(logits.cpu())
        all_train_trues.append(yb.cpu())

train_logits_all = torch.cat(all_train_preds, dim=0)
train_trues_all = torch.cat(all_train_trues, dim=0)

final_train_acc1 = calculate_top_k_accuracy(train_logits_all, train_trues_all, k=1)
final_train_acc3 = calculate_top_k_accuracy(train_logits_all, train_trues_all, k=3)
_, _, train_f1_macro = calculate_macro_metrics(train_logits_all, train_trues_all)

# Complete Validation predictions capture block
val_eval_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
all_val_preds = []
all_val_trues = []

with torch.no_grad():
    for xb, yb in val_eval_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        all_val_preds.append(logits.cpu())
        all_val_trues.append(yb.cpu())

val_logits_all = torch.cat(all_val_preds, dim=0)
val_trues_all = torch.cat(all_val_trues, dim=0)

final_val_acc1 = calculate_top_k_accuracy(val_logits_all, val_trues_all, k=1)
final_val_acc3 = calculate_top_k_accuracy(val_logits_all, val_trues_all, k=3)
_, _, val_f1_macro = calculate_macro_metrics(val_logits_all, val_trues_all)

# NEW: Print per-class classification matrices directly to console
print("\n" + "-"*20 + " SCALED CLASSIFICATION REPORT " + "-"*20)
val_preds_classes = torch.argmax(val_logits_all, dim=1).numpy()
if IS_BINARY:
    print(classification_report(val_trues_all.numpy(), val_preds_classes, target_names=['SKIP (0)', 'CALL/RIICHI (1)'], zero_division=0))
else:
    print(classification_report(val_trues_all.numpy(), val_preds_classes, zero_division=0))

# Final comparative terminal display output
print("\n### METRIC SUMMARY ###")
print(f"Train Dataset Size: {len(X_train)} | Val Dataset Size: {len(X_val)}")
print(f"Total Execution Wall Time: {hours}h {minutes}m {seconds:.2f}s")
print("-" * 50)
print(f"Top-1 Accuracy / Decision Precision Match:")
print(f"  -> Train Accuracy: {final_train_acc1 * 100:.2f}%")
print(f"  -> Val/Test Accuracy: {final_val_acc1 * 100:.2f}%")
print(f"  -> Generalization Gap: {(final_train_acc1 - final_val_acc1) * 100:.2f}%")
print("-" * 50)
print(f"Strategic Macro-F1 Balancer:")
print(f"  -> Train Macro F1: {train_f1_macro * 100:.2f}%")
print(f"  -> Val/Test Macro F1: {val_f1_macro * 100:.2f}%")

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