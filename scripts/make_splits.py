import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.io import save_json
from src.utils.seed import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--val_size", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.labels)
    df["image_id"] = df["image_id"].astype(str)
    ids = df["image_id"].tolist()

    os.makedirs(args.out, exist_ok=True)

    for s in [int(x.strip()) for x in args.seeds.split(",")]:
        set_seed(s)
        train_ids, temp_ids = train_test_split(ids, test_size=(args.test_size + args.val_size), random_state=s, shuffle=True, stratify=df["label"])
        # split temp into val/test
        temp_df = df[df["image_id"].isin(temp_ids)]
        val_ratio = args.val_size / (args.val_size + args.test_size)
        val_ids, test_ids = train_test_split(temp_ids, test_size=(1 - val_ratio), random_state=s, shuffle=True, stratify=temp_df["label"])
        split = {"seed": s, "train": train_ids, "val": val_ids, "test": test_ids}
        save_json(split, os.path.join(args.out, f"seed_{s}.json"))
        print(f"wrote splits: {args.out}/seed_{s}.json | train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

if __name__ == "__main__":
    main()
