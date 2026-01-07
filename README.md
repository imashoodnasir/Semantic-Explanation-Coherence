# semantic explanation coherence (sec) — skin cancer xai evaluation

This repository is a **reproducible python implementation template** for the paper/thesis workflow:

- datasets: **ham10000**, **isic 2019**, **isic 2020**
- baselines: any `timm` model (cnn / convnext / transformer) + your custom checkpoints
- xai: **grad-cam**, **grad-cam++**, **score-cam**, **integrated gradients**, **attention rollout**
- sec: **lesion-level**, **attribute-level**, **weighted sec**
- experiments: in-dataset, cross-dataset, ablation, runtime + memory, statistical testing (anova + tukey)

> note: this repo does not ship datasets. you must download datasets yourself and point configs to your paths.

---

## 1) setup

### 1.1 create env
```bash
python -m venv .venv
source .venv/bin/activate  # linux/mac
# .venv\Scripts\activate  # windows
pip install -r requirements.txt
```

### 1.2 folder layout (expected)
```
data/
  ham10000/
    images/                 # jpg/png
    labels.csv              # columns: image_id,label
    lesion_masks/           # optional: image_id.png (0/255)
    attr_masks/
      asymmetry/
      border/
      pigment_network/
      streaks/
      color/
  isic2019/
    images/
    labels.csv
    lesion_masks/
    attr_masks/...
  isic2020/
    images/
    labels.csv              # binary label for melanoma vs others (0/1)
    lesion_masks/           # optional (can be pseudo)
    attr_masks/...          # optional
splits/
  ham10000/seed_1.json ...  # created by scripts/make_splits.py
outputs/
  checkpoints/
  heatmaps/
  sec/
  tables/
  figures/
```

---

## 2) quick start (smoke test)

### 2.1 create splits (3 seeds)
```bash
python scripts/make_splits.py --dataset ham10000 --labels data/ham10000/labels.csv --out splits/ham10000
python scripts/make_splits.py --dataset isic2019  --labels data/isic2019/labels.csv  --out splits/isic2019
python scripts/make_splits.py --dataset isic2020  --labels data/isic2020/labels.csv  --out splits/isic2020
```

### 2.2 train a baseline (example)
```bash
python scripts/train_classifier.py --config configs/ham10000_resnet50.yaml
```

### 2.3 generate heatmaps (example)
```bash
python scripts/generate_heatmaps.py --config configs/ham10000_resnet50.yaml --split test --xai all
```

### 2.4 compute sec (lesion / attribute / weighted)
```bash
python scripts/compute_sec.py --config configs/ham10000_resnet50.yaml --split test
```

### 2.5 cross-dataset sec (train on ham, eval on isic2020)
```bash
python scripts/cross_dataset_sec.py --src_config configs/ham10000_resnet50.yaml --tgt_config configs/isic2020_resnet50.yaml
```

### 2.6 statistics (anova + tukey) on overall sec
```bash
python scripts/stats_sec.py --table outputs/tables/sec_overall_ham10000_test.csv --out outputs/tables/stats_ham10000.csv
```

---

## 3) configs

All experiments are driven by yaml configs in `configs/`.

Key fields:
- `dataset.name`, `dataset.root`, `dataset.labels_csv`
- `model.name` (timm model), `model.num_classes`
- `train.*` hyperparameters
- `xai.*` parameters (target layer selection, ig steps, etc.)
- `sec.*` parameters (weights, smoothing, etc.)

---

## 4) masks (lesion + attributes)

SEC needs:
- lesion mask: `lesion_masks/{image_id}.png` (0/255)
- attribute masks (optional): `attr_masks/<attr>/{image_id}.png`

If you do not have masks:
- use pseudo lesion masks from your u-net and save them to `lesion_masks/`.
- for attribute masks, you can either annotate or generate approximations.

The code supports missing attribute masks: it will skip missing attributes and report coverage.

---

## 5) outputs

- heatmaps: `outputs/heatmaps/<dataset>/<model>/<xai>/<image_id>.npy`
- sec per-image + aggregates: `outputs/sec/...`
- latex-ready tables: `outputs/tables/*.csv` + `*.tex`

---

## 6) citation

If you use this repo in a thesis/paper, cite your SEC work accordingly.

---

## 7) troubleshooting

- grad-cam layer: ensure `xai.target_layer` points to a conv layer (cnn). for transformers, use attention rollout.
- isic2020 is binary by default. set `model.num_classes=2` and label mapping accordingly.

---

## license
MIT (template code)
