import zipfile
from pathlib import Path

ZIP_PATH = Path("data/raw/tenhou-4-player-riichi-mahjong-dataset.zip")
OUT_DIR = Path("data/raw")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    print("Files inside zip:")
    
    for f in z.namelist():
        print(f)

    print("\nExtracting datasets_positive.db ...")

    z.extract("datasets_positive.db", OUT_DIR)

print("\nDone.")