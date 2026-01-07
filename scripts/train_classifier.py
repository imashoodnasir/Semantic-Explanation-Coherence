import argparse, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.io import load_yaml, load_json
from src.utils.seed import set_seed
from src.datasets.transforms import build_transforms
from src.datasets.dermoscopy import DermoscopyDataset
from src.models.build import build_model
from src.metrics.classification import compute_metrics
from src.utils.paths import ensure_dir

def get_optimizer(name, params, lr, weight_decay):
    if name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["train"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_json(cfg["dataset"]["split_json"])
    img_size = int(cfg["dataset"]["image_size"])
    num_classes = int(cfg["model"]["num_classes"])

    tfm_train = build_transforms(img_size, train=True)
    tfm_eval = build_transforms(img_size, train=False)

    ds_train = DermoscopyDataset(cfg["dataset"]["root"], cfg["dataset"]["images_dir"], cfg["dataset"]["labels_csv"], split["train"], tfm_train, num_classes)
    ds_val = DermoscopyDataset(cfg["dataset"]["root"], cfg["dataset"]["images_dir"], cfg["dataset"]["labels_csv"], split["val"], tfm_eval, num_classes)

    dl_train = DataLoader(ds_train, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=int(cfg["train"]["num_workers"]), pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"]["num_workers"]), pin_memory=True)

    model = build_model(cfg["model"]["name"], num_classes=num_classes, pretrained=bool(cfg["model"]["pretrained"]))
    model.to(device)

    opt = get_optimizer(cfg["train"]["optimizer"], model.parameters(), float(cfg["train"]["lr"]), float(cfg["train"]["weight_decay"]))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))

    out_dir = ensure_dir("outputs/checkpoints")
    best_path = os.path.join(out_dir, f"{cfg['dataset']['name']}_{cfg['model']['name']}_seed{cfg['train']['seed']}.pt")
    best_score = -1.0

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        for x, y, _ in tqdm(dl_train, desc=f"train epoch {epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"])):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # val
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x, y, _ in tqdm(dl_val, desc=f"val epoch {epoch}"):
                x = x.to(device, non_blocking=True)
                logits = model(x)
                prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
                pred = np.argmax(prob, axis=1)
                y_true.extend(y.numpy().tolist())
                y_pred.extend(pred.tolist())
                y_prob.extend(prob.tolist())

        m = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
        score = m["macro_f1"] if not np.isnan(m["macro_f1"]) else m["accuracy"]
        print(f"epoch={epoch} | val metrics: {m}")

        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "config": cfg}, best_path)
            print(f"saved best checkpoint -> {best_path}")

    print("done. best checkpoint:", best_path)

if __name__ == "__main__":
    main()
