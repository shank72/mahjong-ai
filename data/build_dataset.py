import numpy as np
from tqdm import tqdm
from lazy_dataloader import MahjongDataset
from feature_encoder import encode


def build_dataset(db_path, table="Discard", limit=None, out_prefix="dataset"):
    """
    Converts SQLite dataset → ML tensors
    """

    dataset = MahjongDataset(
        db_path=db_path,
        table=table,
        limit=limit
    )

    X = []
    y = []

    print(f"[INFO] Building dataset from table: {table}")

    for i in tqdm(range(len(dataset))):
        record = dataset[i]

        try:
            vec, label = encode(record)

            X.append(vec)
            y.append(label)

        except Exception as e:
            # skip bad rows safely
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print("\n[INFO] Dataset built:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save(f"{out_prefix}_X.npy", X)
    np.save(f"{out_prefix}_y.npy", y)

    print(f"[DONE] Saved to {out_prefix}_X.npy and {out_prefix}_y.npy")

    dataset.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default="Discard")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="dataset")

    args = parser.parse_args()

    build_dataset(
        db_path=args.db,
        table=args.table,
        limit=args.limit,
        out_prefix=args.out
    )

    # python data/build_dataset.py --db data/raw/datasets_positive.db --limit 200000
    