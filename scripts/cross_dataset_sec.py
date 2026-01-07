import argparse, os
import numpy as np
import pandas as pd
from src.utils.io import load_yaml
from src.utils.paths import join, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_config", required=True, help="trained-on dataset config (used to locate checkpoint + model name)")
    ap.add_argument("--tgt_config", required=True, help="target dataset config (used to locate target splits + masks)")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    args = ap.parse_args()

    src = load_yaml(args.src_config)
    tgt = load_yaml(args.tgt_config)

    # assumes you generated target heatmaps using the SRC model checkpoint on target images
    # recommended workflow:
    # 1) set tgt.model.checkpoint_path = src checkpoint
    # 2) run generate_heatmaps.py with tgt_config
    # 3) run compute_sec.py with tgt_config
    # this script just provides a consistent naming and a stability table scaffold.

    out_dir = ensure_dir("outputs/tables")
    note = f"{src['dataset']['name']}_to_{tgt['dataset']['name']}_{args.split}"
    print("cross-dataset workflow scaffold:", note)
    print("1) set tgt model checkpoint to src checkpoint")
    print("2) run generate_heatmaps on target config")
    print("3) run compute_sec on target config to produce overall sec table")
    print("done.")

if __name__ == "__main__":
    main()
