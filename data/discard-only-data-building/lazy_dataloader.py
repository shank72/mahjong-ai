import sqlite3
import gzip
import json
from pathlib import Path

from torch.utils.data import Dataset


class MahjongDataset(Dataset):
    """
    PyTorch Dataset for Tenhou Mahjong data.

    Each row in the SQLite DB contains:
        - gzip-compressed JSON blob

    We:
        SQLite row
            -> gzip decompress
            -> json parse
            -> Python dict
    """

    def __init__(self, db_path, table="Discard", limit=None):
        self.db_path = Path(db_path)
        self.table = table

        # open sqlite connection
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()

        # count rows
        query = f"SELECT COUNT(*) FROM {self.table}"
        self.cursor.execute(query)

        self.length = self.cursor.fetchone()[0]

        if limit is not None:
            self.length = min(self.length, limit)

        print(f"[{self.table}] loaded with {self.length:,} rows")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns ONE parsed Mahjong state dict.
        """

        # SQLite rows are 1-indexed in many datasets
        row_id = idx + 1

        query = f"""
        SELECT *
        FROM {self.table}
        WHERE rowid = ?
        """

        self.cursor.execute(query, (row_id,))

        row = self.cursor.fetchone()

        if row is None:
            raise IndexError(f"Missing row {row_id}")

        # usually compressed blob is column 0 or 1
        #compressed = row[0]
        compressed = row[1]

        # decompress gzip
        decompressed = gzip.decompress(compressed)

        # bytes -> string
        json_str = decompressed.decode("utf-8")

        # string -> dict
        data = json.loads(json_str)

        return data

    def close(self):
        self.conn.close()