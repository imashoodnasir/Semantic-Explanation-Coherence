from typing import Dict, List, Optional, Tuple
import numpy as np

def lesion_sec(E: np.ndarray, L: np.ndarray, eps: float = 1e-8) -> float:
    # E: HxW mass-normalized
    return float((E * L).sum() / (E.sum() + eps))

def attribute_sec(E: np.ndarray, attr_masks: Dict[str, np.ndarray], eps: float = 1e-8) -> float:
    if len(attr_masks) == 0:
        return float("nan")
    union = np.zeros_like(E, dtype=np.float32)
    for m in attr_masks.values():
        union = np.maximum(union, m.astype(np.float32))
    return float((E * union).sum() / (E.sum() + eps))

def weighted_sec(E: np.ndarray, attr_masks: Dict[str, np.ndarray], weights: Dict[str, float], eps: float = 1e-8) -> float:
    if len(attr_masks) == 0:
        return float("nan")
    # normalize weights over available attrs
    ws = []
    for k in attr_masks.keys():
        ws.append(max(float(weights.get(k, 0.0)), 0.0))
    s = sum(ws)
    if s <= 0:
        ws = [1.0 for _ in ws]
        s = float(len(ws))
    ws = [w / s for w in ws]

    score = 0.0
    for (a, m), w in zip(attr_masks.items(), ws):
        score += w * float((E * m).sum() / (E.sum() + eps))
    return float(score)

def overall_sec(sec_les: float, sec_attr: float, sec_w: float) -> float:
    # simple mean over available components
    vals = [v for v in [sec_les, sec_attr, sec_w] if not (np.isnan(v) or np.isinf(v))]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))
