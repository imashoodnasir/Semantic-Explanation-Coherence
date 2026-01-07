import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

def _read_mask(path: str, size: int = 224, thresh: float = 0.5) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    m = (m.astype(np.float32) / 255.0)
    m = (m >= thresh).astype(np.float32)
    return m

def load_lesion_mask(root: str, image_id: str, size: int = 224, thresh: float = 0.5) -> Optional[np.ndarray]:
    p = os.path.join(root, f"{image_id}.png")
    if not os.path.exists(p):
        return None
    return _read_mask(p, size=size, thresh=thresh)

def load_attribute_masks(attr_root: str, attrs: List[str], image_id: str, size: int = 224, thresh: float = 0.5) -> Dict[str, np.ndarray]:
    out = {}
    for a in attrs:
        p = os.path.join(attr_root, a, f"{image_id}.png")
        if os.path.exists(p):
            out[a] = _read_mask(p, size=size, thresh=thresh)
    return out
