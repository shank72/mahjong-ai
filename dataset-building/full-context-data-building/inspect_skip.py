import json
from lazy_dataloader import MahjongDataset

# Point this to your actual database file path
DB_PATH = "data-extraction/raw/datasets_positive.db" 

dataset = MahjongDataset(db_path=DB_PATH, table="Skip", limit=5)

print("\n--- CRITICAL DATA INSPECTION ---")
for i in range(min(3, len(dataset))):
    print(f"\n[ROW {i}] Raw Keys available in this record:")
    record = dataset[i]
    print(list(record.keys()))
    
    if "valid_actions" in record:
        print(f"Content of 'valid_actions':")
        print(json.dumps(record["valid_actions"][:3], indent=2))
    else:
        print("'valid_actions' key DOES NOT EXIST in this table's records!")
        
    if "action_idx" in record:
        print(f"'action_idx' value: {record['action_idx']}")

dataset.close()