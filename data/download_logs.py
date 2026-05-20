from pathlib import Path

db_path = Path("data/raw/tenhou-4-player-riichi-mahjong-dataset.db")

print("Exists:", db_path.exists())
print("Absolute path:", db_path.resolve())
print("Size (MB):", db_path.stat().st_size / 1024 / 1024)



from pathlib import Path

root = Path("data/raw")

for p in root.iterdir():
    size_mb = p.stat().st_size / 1024 / 1024
    print(f"{p.name:<60} {size_mb:.2f} MB")