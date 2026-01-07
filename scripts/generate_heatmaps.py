import argparse, os, time, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
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
from src.utils.paths import ensure_dir, join

def save_npy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr.astype(np.float32))

def upsample_to_224(mask_patch: np.ndarray, size: int = 224):
    return cv2.resize(mask_patch, (size, size), interpolation=cv2.INTER_CUBIC)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--xai", default="all", help="comma list: gradcam,gradcampp,scorecam,ig,attention_rollout or all")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_json(cfg["dataset"]["split_json"])
    ids = split[args.split]

    img_size = int(cfg["dataset"]["image_size"])
    tfm = build_transforms(img_size, train=False)
    ds = DermoscopyDataset(cfg["dataset"]["root"], cfg["dataset"]["images_dir"], cfg["dataset"]["labels_csv"], ids, tfm, int(cfg["model"]["num_classes"]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = build_model(cfg["model"]["name"], num_classes=int(cfg["model"]["num_classes"]), pretrained=bool(cfg["model"]["pretrained"]))
    model = load_checkpoint(model, cfg["model"].get("checkpoint_path",""))
    model.to(device).eval()

    methods = cfg["xai"]["methods"] if args.xai == "all" else [m.strip() for m in args.xai.split(",")]
    out_root = ensure_dir(join("outputs","heatmaps", cfg["dataset"]["name"], cfg["model"]["name"]))

    for x, y, image_id in tqdm(dl, desc="heatmaps"):
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
        for m in methods:
            if m in ["gradcam","gradcampp","scorecam"]:
                h = run_cam(model, x, pred, method=m, target_layer=str(cfg["xai"]["target_layer"]))
                h = softmax_spatial(h) if bool(cfg["sec"]["heatmap_softmax"]) else h
            elif m == "ig":
                h = run_ig(model, x, pred, steps=int(cfg["xai"]["ig_steps"]), baseline=str(cfg["xai"]["ig_baseline"]))
                h = softmax_spatial(h) if bool(cfg["sec"]["heatmap_softmax"]) else h
            elif m == "attention_rollout":
                # transformer only; returns patch-grid, then upsample
                try:
                    patch = attention_rollout(model, x)
                    h = upsample_to_224(patch, size=img_size)
                    h = normalize_heatmap(h)
                    h = softmax_spatial(h) if bool(cfg["sec"]["heatmap_softmax"]) else h
                except Exception:
                    continue
            else:
                continue

            save_path = join(out_root, m, f"{image_id[0]}.npy")
            save_npy(save_path, h)

if __name__ == "__main__":
    main()
