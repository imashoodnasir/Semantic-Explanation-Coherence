import numpy as np
import torch

def normalize_heatmap(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    h = np.maximum(h, 0.0)
    h = (h - h.min()) / (h.max() - h.min() + eps)
    return h

def softmax_spatial(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # stable softmax over HxW
    x = h.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    denom = np.sum(ex) + eps
    return (ex / denom).astype(np.float32)
