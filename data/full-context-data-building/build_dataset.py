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
    
    if table == "Discard":
        # Target: Which of the 34 tile types was tossed
        return tile_to_34(chosen_action["tiles"][0])
        
    elif table in ["Pon", "Chi", "DaiMinKan"]:
        # Target: Binary decision. 
        # In Tenhou schema, Pass/Skip actions possess a "type" value of 0.
        action_type = chosen_action.get("type", 0)
        return 0 if action_type == 0 else 1
        
    elif table == "Riichi":
        # Target: Binary decision. Riichi type is declaration action.
        action_type = chosen_action.get("type", 0)
        return 1 if action_type == 7 else 0
        
    else:
        raise ValueError(f"Target logic mapping undefined for table type: {table}")

def build_dataset(db_path, table="Discard", limit=None, out_prefix="dataset"):
    dataset = MahjongDataset(db_path=db_path, table=table, limit=limit)
    X, y = [], []

    print(f"[INFO] Compiling tensor matrices from table: {table}")

    for i in tqdm(range(len(dataset))):
        try:
            record = dataset[i]
            
            # 1. Generate the shared situational geometry
            vec = encode_state_vector(record)
            
            # 2. Safely capture target mapping
            label = get_target_label(record, table)

            X.append(vec)
            y.append(label)
        except Exception as e:
            # Silently drop non-standard formatting variances safely
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\n[SUCCESS] Completed {table} Engine Generation:")
    print("-> Features shape:", X.shape)
    print("-> Labels shape:  ", y.shape)

    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)
    print(f"[SAVED] File sequence target: {out_prefix}_X.npy\n")
    dataset.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to datasets_positive.db")
    parser.add_argument("--table", default="Discard", help="Discard, Pon, Chi, Riichi")
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