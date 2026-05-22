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
        
    # Define a clean mapping of table names to their expected active Tenhou IDs
    table_action_mapping = {
        "Chi": 1,
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
        for i in tqdm(range(len(dataset))):
            try:
                record = dataset[i]
                X.append(encode_state_vector(record))
                y.append(get_target_label(record, table))
            except Exception:
                continue
        dataset.close()
    else:
        # --- BALANCED BINARY MELDMATH ENGINE ---
        print(f"[INFO] Extracting balanced binary sets for: {table.upper()}")
        
        # 1. Gather the POSITIVE instances (The actual action calls)
        print(f" -> Mining {half_limit} 'Calls' from table: {table}")
        pos_dataset = MahjongDataset(db_path=db_path, table=table, limit=half_limit)
        for i in tqdm(range(len(pos_dataset))):
            try:
                record = pos_dataset[i]
                X.append(encode_state_vector(record))
                y.append(1) # Forced true: every row in these tables is an action execution
            except Exception:
                continue
        pos_dataset.close()
        
        # 2. Gather the NEGATIVE instances (The pass/skips from the central Skip table)
        print(f" -> Mining {half_limit} valid 'Skips' from table: Skip")
        
        # Request a higher lookahead limit because the Skip table contains mix of Pon/Chi/Kan skips
        skip_dataset = MahjongDataset(db_path=db_path, table="Skip", limit=half_limit * 6)
        skips_collected = 0
        
        for i in tqdm(range(len(skip_dataset))):
            if skips_collected >= half_limit:
                break
            try:
                record = skip_dataset[i]
                
                # Check what action was skipped. We map the target label logic 
                # against the structural record to make sure it matches the table we want
                action_idx = record["action_idx"]
                chosen_action = record["valid_actions"][action_idx]
                
                # Verify that this skip row matches the specific action context we are training
                is_valid_skip_context = False
                action_type = chosen_action.get("type", 0)
                
                # In Tenhou frameworks: Type 0 = Pass/Skip. We examine what ELSE was valid.
                valid_types = [act.get("type") for act in record["valid_actions"]]
                
                if table == "Chi" and 1 in valid_types:       # Type 1 = Chi option was present
                    is_valid_skip_context = True
                elif table == "Pon" and 3 in valid_types:     # Type 3 = Pon option was present
                    is_valid_skip_context = True
                elif table == "Riichi" and 7 in valid_types:  # Type 7 = Riichi option was present
                    is_valid_skip_context = True
                elif table == "DaiMinKan" and 4 in valid_types: # Type 4 = Open Kan option present
                    is_valid_skip_context = True
                elif table == "AnKan" and 5 in valid_types:     # Type 5 = Closed Kan option present
                    is_valid_skip_context = True
                elif table == "ShouMinKan" and 6 in valid_types: # Type 6 = Added Kan option present
                    is_valid_skip_context = True

                if is_valid_skip_context:
                    X.append(encode_state_vector(record))
                    y.append(0) # It's a verified passive skip
                    skips_collected += 1
            except Exception:
                continue
        skip_dataset.close()

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