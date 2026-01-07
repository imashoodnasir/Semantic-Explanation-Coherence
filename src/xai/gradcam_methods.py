from typing import Optional, List
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .common import normalize_heatmap

def _get_target_layers(model: torch.nn.Module, target_layer: str):
    # user provides a string like "layer4" for resnet-like models.
    if hasattr(model, target_layer):
        return [getattr(model, target_layer)]
    # fallback: last conv2d found
    last_conv = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No conv layer found for grad-cam. Set xai.target_layer correctly.")
    return [last_conv]

def run_cam(model: torch.nn.Module, inp: torch.Tensor, target_class: int, method: str, target_layer: str) -> np.ndarray:
    model.eval()
    target_layers = _get_target_layers(model, target_layer)
    targets = [ClassifierOutputTarget(target_class)]
    if method == "gradcam":
        cam = GradCAM(model=model, target_layers=target_layers)
    elif method == "gradcampp":
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    elif method == "scorecam":
        cam = ScoreCAM(model=model, target_layers=target_layers)
    else:
        raise ValueError("method must be gradcam | gradcampp | scorecam")
    grayscale = cam(input_tensor=inp, targets=targets)[0]  # HxW
    return normalize_heatmap(grayscale)
