import argparse, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import load_yaml, load_json
from src.datasets.transforms import build_transforms
from src.datasets.dermoscopy import DermoscopyDataset
from src.models.build import build_model, load_checkpoint
from src.xai.gradcam_methods import run_cam
from src.xai.integrated_gradients import run_ig
from src.xai.attention_rollout import attention_rollout
from src.xai.common import softmax_spatial, normalize_heatmap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_json(cfg["dataset"]["split_json"])
    ids = split[args.split][:args.n]

    tfm = build_transforms(int(cfg["dataset"]["image_size"]), train=False)
    ds = DermoscopyDataset(cfg["dataset"]["root"], cfg["dataset"]["images_dir"], cfg["dataset"]["labels_csv"], ids, tfm, int(cfg["model"]["num_classes"]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    model = build_model(cfg["model"]["name"], int(cfg["model"]["num_classes"]), pretrained=bool(cfg["model"]["pretrained"]))
    model = load_checkpoint(model, cfg["model"].get("checkpoint_path",""))
    model.to(device).eval()

    rows = []
    for method in cfg["xai"]["methods"]:
        times = []
        mems = []
        for x, _, _ in tqdm(dl, desc=f"profile {method}"):
            x = x.to(device)
            with torch.no_grad():
                pred = int(torch.argmax(model(x), dim=1).item())

            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            if method in ["gradcam","gradcampp","scorecam"]:
                h = run_cam(model, x, pred, method=method, target_layer=str(cfg["xai"]["target_layer"]))
            elif method == "ig":
                h = run_ig(model, x, pred, steps=int(cfg["xai"]["ig_steps"]), baseline=str(cfg["xai"]["ig_baseline"]))
            elif method == "attention_rollout":
                try:
                    patch = attention_rollout(model, x)
                    h = patch
                except Exception:
                    continue
            else:
                continue
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if torch.cuda.is_available():
                mems.append(torch.cuda.max_memory_allocated() / (1024**3))

        if len(times) > 0:
            rows.append({
                "dataset": cfg["dataset"]["name"],
                "xai": method,
                "time_sec_per_image": float(np.mean(times)),
                "gpu_mem_gb_peak": float(np.mean(mems)) if len(mems) > 0 else float("nan"),
                "n": len(times),
            })

    import pandas as pd
    os.makedirs("outputs/tables", exist_ok=True)
    out = f"outputs/tables/profile_{cfg['dataset']['name']}_{cfg['model']['name']}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote:", out)

if __name__ == "__main__":
    main()
