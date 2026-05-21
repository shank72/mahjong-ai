from data.lazy_dataloader import MahjongDataset
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "raw" / "datasets_positive.db"

print("DB path:", DB_PATH)
print("Exists:", DB_PATH.exists())

dataset = MahjongDataset(
    db_path=DB_PATH,
    table="Discard",
    limit=5
)

print()

sample = dataset[0]

print("Keys:")
print(sample.keys())

print("\nHand:")
print(sample["hand_tiles"])

print("\nChosen action:")
print(sample["action_idx"])

print("\nValid actions:")
print(sample["valid_actions"][:3])

chosen_action = sample["valid_actions"][sample["action_idx"]]

print("\nChosen action object:")
print(chosen_action)

dataset.close()