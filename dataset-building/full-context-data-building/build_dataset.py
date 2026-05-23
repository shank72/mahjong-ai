import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
import argparse

from lazy_dataloader import MahjongDataset
from feature_encoder import encode_state_vector, tile_to_34

def get_target_label(record: dict, table: str) -> int:
    """
    Dynamically parses the proper ML target label based on table mechanics.
    """
    action_idx = record["action_idx"]
    chosen_action = record["valid_actions"][action_idx]
    action_type = chosen_action.get("type", 0)
    
    if table == "Discard":
        return tile_to_34(chosen_action["tiles"][0])
        
    # FIX: Updated to map precisely to your custom database schema IDs
    table_action_mapping = {
        "Chi": 2,
        "Pon": 3,
        "DaiMinKan": 4,
        "AnKan": 5,
        "ShouMinKan": 6,
        "Riichi": 7
    }
    
    if table in table_action_mapping:
        expected_type = table_action_mapping[table]
        # It's a Call (1) if the action matches the table type, otherwise it's a Skip (0)
        return 1 if action_type == expected_type else 0
    else:
        raise ValueError(f"Target logic mapping undefined for table type: {table}")

def build_dataset(db_path, table="Discard", limit=None, out_prefix="dataset"):
    is_binary = table != "Discard"
    half_limit = limit // 2 if limit is not None else float('inf')
    
    X, y = [], []

    if not is_binary:
        # --- STANDARD MULTI-CLASS DISCARD TRACK ---
        print(f"[INFO] Compiling multi-class engine matrices from table: {table}")
        dataset = MahjongDataset(db_path=db_path, table=table, limit=limit)
        
        pbar = tqdm(total=dataset.length, desc=f"Processing {table}")
        for i in range(dataset.length):
            try:
                record = dataset[i]
                if record is None:
                    break
                X.append(encode_state_vector(record))
                y.append(get_target_label(record, table))
                pbar.update(1)
            except (IndexError, KeyError):
                break
            except Exception:
                continue
        pbar.close()
        dataset.close()
    else:
        # --- BALANCED BINARY MELDMATH ENGINE ---
        print(f"[INFO] Extracting balanced binary sets for: {table.upper()}")
        
        # 1. Gather the POSITIVE instances (The actual action calls)
        print(f" -> Mining {half_limit} 'Calls' from table: {table}")
        pos_dataset = MahjongDataset(db_path=db_path, table=table, limit=half_limit)
        
        pbar_pos = tqdm(total=pos_dataset.length, desc="Collecting Calls")
        for i in range(pos_dataset.length):
            try:
                record = pos_dataset[i]
                if record is None:
                    break
                X.append(encode_state_vector(record))
                y.append(1)  # Forced true: every row in these tables is an action execution
                pbar_pos.update(1)
            except (IndexError, KeyError):
                break
            except Exception:
                continue
        pbar_pos.close()
        pos_dataset.close()
        
        # 2. Gather the NEGATIVE instances (The pass/skips from the central Skip table)
        print(f" -> Mining {half_limit} valid 'Skips' from table: Skip")
        
        skip_dataset = MahjongDataset(db_path=db_path, table="Skip", limit=None)
        skips_collected = 0
        
        pbar_neg = tqdm(total=half_limit, desc="Collecting Skips")
        
        for i in range(skip_dataset.length):
            if skips_collected >= half_limit:
                break
            try:
                record = skip_dataset[i]
                if record is None:
                    break
                
                # FIX: Match structural context using raw integer comparisons matching your logs
                valid_types = [act.get("type") for act in record["valid_actions"]]
                is_valid_skip_context = False

                # Adjusted filtering blocks to reflect verified type positions
                if table == "Chi" and 2 in valid_types:
                    is_valid_skip_context = True
                elif table == "Pon" and 3 in valid_types:
                    is_valid_skip_context = True
                elif table == "Riichi" and 7 in valid_types:
                    is_valid_skip_context = True
                elif table == "DaiMinKan" and 4 in valid_types:
                    is_valid_skip_context = True
                elif table == "AnKan" and 5 in valid_types:
                    is_valid_skip_context = True
                elif table == "ShouMinKan" and 6 in valid_types:
                    is_valid_skip_context = True

                if is_valid_skip_context:
                    X.append(encode_state_vector(record))
                    y.append(0)  # It's a verified passive skip
                    skips_collected += 1
                    pbar_neg.update(1)
            except (IndexError, KeyError):
                break
            except Exception:
                continue
        pbar_neg.close()
        skip_dataset.close()

    # --- AUTOMATED DATA BALANCE INSURANCE ---
    if is_binary:
        y_temp = np.array(y)
        pos_count = np.sum(y_temp == 1)
        neg_count = np.sum(y_temp == 0)
        
        if pos_count != neg_count:
            print(f"[WARNING] Asymmetric class extraction detected (Calls: {pos_count} | Skips: {neg_count})")
            min_samples = min(pos_count, neg_count)
            
            if min_samples == 0:
                print("[CRITICAL ERROR] One of your data buckets is completely empty! Check types.")
                return
                
            print(f" -> Automatically balancing dataset down to {min_samples} rows per class.")
            X_pos_final = [X[i] for i in range(len(y)) if y[i] == 1][:min_samples]
            X_neg_final = [X[i] for i in range(len(y)) if y[i] == 0][:min_samples]
            
            X = X_pos_final + X_neg_final
            y = [1] * min_samples + [0] * min_samples

    # --- SHUFFLE & EXPORT ---
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    if len(y) > 0 and is_binary:
        shuffle_indices = np.random.permutation(len(y))
        X = X[shuffle_indices]
        y = y[shuffle_indices]

    # Save logic and reporting outputs
    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[SUCCESS] Completed {table} Engine Generation:")
    print("-> Features shape:", X.shape)
    print("-> Labels shape:  ", y.shape)
    if is_binary and len(y) > 0:
        print(f"-> Positive Class Ratio: {np.mean(y) * 100:.2f}%")

    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)
    print(f"[SAVED] File sequence target: {out_prefix}_X.npy\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to datasets_positive.db")
    parser.add_argument("--table", default="Discard", help="Discard, Pon, Chi, Riichi, DaiMinKan, AnKan, ShouMinKan")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="dataset")

    args = parser.parse_args()
    start_time = time.perf_counter()

    build_dataset(
        db_path=args.db,
        table=args.table,
        limit=args.limit,
        out_prefix=args.out
    )

    print(f"Execution runtime performance: {time.perf_counter() - start_time:.2f} seconds")

'''
# 1. Core Multi-Class Engine (Needs maximum volume)
python dataset-building/full-context-data-building/build_dataset.py --db data-extraction/raw/datasets_positive.db --table Discard --limit 500000 --out datasets/full-context/discard

# 2. Primary Meld Engines (High contextual nuance)
python dataset-building/full-context-data-building/build_dataset.py --db data-extraction/raw/datasets_positive.db --table Chi --limit 100000 --out datasets/full-context/chi
python dataset-building/full-context-data-building/build_dataset.py --db data-extraction/raw/datasets_positive.db --table Pon --limit 100000 --out datasets/full-context/pon

# 3. Special Strategic State
python dataset-building/full-context-data-building/build_dataset.py --db data-extraction/raw/datasets_positive.db --table Riichi --limit 50000 --out datasets/full-context/riichi

# 4. Rare Kan Mechanics (Capped cleanly at the data floor boundary)
python dataset-building/full-context-data-building/build_dataset.py --db data-extraction/raw/datasets_positive.db --table DaiMinKan --limit 20000 --out datasets/full-context/daiminkan
python dataset-building/full-context-data-building/build_dataset.py --db data-extraction/raw/datasets_positive.db --table AnKan --limit 20000 --out datasets/full-context/ankan
python dataset-building/full-context-data-building/build_dataset.py --db data-extraction/raw/datasets_positive.db --table ShouMinKan --limit 20000 --out datasets/full-context/shouminkan
'''