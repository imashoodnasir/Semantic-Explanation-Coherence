import argparse, os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.io import load_yaml, load_json
from src.utils.paths import ensure_dir, join
from src.sec.masks import load_lesion_mask, load_attribute_masks
from src.sec.compute import lesion_sec, attribute_sec, weighted_sec, overall_sec

def load_heatmap(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    split = load_json(cfg["dataset"]["split_json"])
    ids = split[args.split]

    heat_root = join("outputs","heatmaps", cfg["dataset"]["name"], cfg["model"]["name"])
    out_dir = ensure_dir(join("outputs","tables"))
    out_path = join(out_dir, f"sec_overall_{cfg['dataset']['name']}_{args.split}.csv")

    lesion_dir = join(cfg["dataset"]["root"], cfg["dataset"]["lesion_masks_dir"])
    attr_dir = join(cfg["dataset"]["root"], cfg["dataset"]["attr_masks_dir"])
    attrs = cfg["sec"]["attributes"]
    weights = cfg["sec"]["weights"]

    rows = []
    for image_id in tqdm(ids, desc="sec"):
        L = load_lesion_mask(lesion_dir, image_id, size=int(cfg["dataset"]["image_size"]), thresh=float(cfg["sec"]["lesion_threshold"]))
        attr_masks = load_attribute_masks(attr_dir, attrs, image_id, size=int(cfg["dataset"]["image_size"]), thresh=float(cfg["sec"]["attr_threshold"]))

        for m in cfg["xai"]["methods"]:
            p = join(heat_root, m, f"{image_id}.npy")
            if not os.path.exists(p):
                continue
            E = load_heatmap(p)
            sec_les = lesion_sec(E, L) if L is not None else float("nan")
            sec_attr = attribute_sec(E, attr_masks)
            sec_w = weighted_sec(E, attr_masks, weights)
            sec_all = overall_sec(sec_les, sec_attr, sec_w)
            rows.append({
                "dataset": cfg["dataset"]["name"],
                "split": args.split,
                "image_id": image_id,
                "xai": m,
                "lesion_sec": sec_les,
                "attribute_sec": sec_attr,
                "weighted_sec": sec_w,
                "overall_sec": sec_all,
                "num_attr_masks": len(attr_masks),
                "has_lesion_mask": int(L is not None),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print("wrote:", out_path)

    # aggregate summary
    if len(df) > 0:
        summary = df.groupby(["dataset","split","xai"], as_index=False)[["lesion_sec","attribute_sec","weighted_sec","overall_sec"]].mean()
        summary_path = out_path.replace(".csv","_mean.csv")
        summary.to_csv(summary_path, index=False)
        print("wrote:", summary_path)

if __name__ == "__main__":
    main()
