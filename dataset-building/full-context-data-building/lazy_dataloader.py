import sqlite3
import gzip
import json
from pathlib import Path
from torch.utils.data import Dataset

class MahjongDataset(Dataset):
    """
    PyTorch Dataset for Tenhou Mahjong data.
    Optimized for maximum extraction speed using cached valid row IDs.
    """

    def __init__(self, db_path, table="Discard", limit=None):
        self.db_path = Path(db_path)
        self.table = table

        # Open sqlite connection
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()

        # ⚡ OPTIMIZATION: Fetch all valid, existing rowids for this table into RAM once.
        # This bypasses rowid gaps entirely while preserving indexed lookup speed.
        print(f"[{self.table}] Indexing valid rows... (Please wait a brief moment)")
        
        if limit is not None:
            query = f"SELECT rowid FROM {self.table} LIMIT ?"
            self.cursor.execute(query, (limit,))
        else:
            query = f"SELECT rowid FROM {self.table}"
            self.cursor.execute(query)
            
        # Store rowids in a flat Python list
        self.valid_ids = [row[0] for row in self.cursor.fetchall()]
        self.length = len(self.valid_ids)

        print(f"[{self.table}] Loaded and verified {self.length:,} active rows.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns ONE parsed Mahjong state dict using an exact, indexed rowid mapping.
        """
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds for dataset length {self.length}")

        # Map our logical 0-indexed loop position directly to the true database rowid
        true_row_id = self.valid_ids[idx]

        # Use the primary key index (blazing fast O(1)/O(log N) lookup)
        query = f"""
        SELECT *
        FROM {self.table}
        WHERE rowid = ?
        """

        self.cursor.execute(query, (true_row_id,))
        row = self.cursor.fetchone()

        if row is None:
            raise IndexError(f"Internal Error: Rowid {true_row_id} cached but missing on disk.")

        # Compressed blob lives in column 1
        compressed = row[1]

        # Decompress gzip
        decompressed = gzip.decompress(compressed)

        # Bytes -> string -> dict
        json_str = decompressed.decode("utf-8")
        data = json.loads(json_str)

        return data

    def close(self):
        self.conn.close()