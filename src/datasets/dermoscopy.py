import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class DermoscopyDataset(Dataset):
    def __init__(
        self,
        root: str,
        images_dir: str,
        labels_csv: str,
        split_ids: List[str],
        transform=None,
        num_classes: int = 2,
    ):
        self.root = root
        self.images_dir = os.path.join(root, images_dir)
        self.df = pd.read_csv(labels_csv)
        self.df["image_id"] = self.df["image_id"].astype(str)
        self.df = self.df[self.df["image_id"].isin(set(split_ids))].reset_index(drop=True)
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.df)

    def _find_image_path(self, image_id: str) -> str:
        # tries common extensions
        for ext in [".jpg",".jpeg",".png",".JPG",".PNG",".JPEG"]:
            p = os.path.join(self.images_dir, image_id + ext)
            if os.path.exists(p):
                return p
        # sometimes image_id already includes extension
        p2 = os.path.join(self.images_dir, image_id)
        if os.path.exists(p2):
            return p2
        raise FileNotFoundError(f"image not found for id={image_id} in {self.images_dir}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label = int(row["label"])
        img_path = self._find_image_path(image_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = img
        return x, label, image_id
