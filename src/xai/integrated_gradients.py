import numpy as np
import torch
from captum.attr import IntegratedGradients
from .common import normalize_heatmap

def run_ig(model: torch.nn.Module, inp: torch.Tensor, target_class: int, steps: int = 50, baseline: str = "zero") -> np.ndarray:
    model.eval()
    ig = IntegratedGradients(model)
    if baseline == "zero":
        base = torch.zeros_like(inp)
    else:
        base = torch.zeros_like(inp)

    attributions = ig.attribute(inp, baselines=base, target=target_class, n_steps=steps)
    attr = attributions.squeeze(0).detach().cpu().numpy()  # CxHxW
    h = np.mean(np.abs(attr), axis=0)  # HxW
    return normalize_heatmap(h)
