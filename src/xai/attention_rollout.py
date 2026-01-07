import math
from typing import List
import numpy as np
import torch
from .common import normalize_heatmap

def attention_rollout(model: torch.nn.Module, inp: torch.Tensor, head_fusion: str = "mean") -> np.ndarray:
    # timm ViT/DeiT: model.blocks each has blk.attn
    attn_mats: List[torch.Tensor] = []

    def hook_fn(module, input, output):
        # output may vary; we try to catch attention probs if available
        if isinstance(output, torch.Tensor):
            attn_mats.append(output.detach())
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            attn_mats.append(output[0].detach())

    hooks = []
    if not hasattr(model, "blocks"):
        raise RuntimeError("attention rollout requires transformer model with .blocks")
    for blk in model.blocks:
        # often blk.attn.attn_drop receives attention weights
        if hasattr(blk.attn, "attn_drop"):
            hooks.append(blk.attn.attn_drop.register_forward_hook(hook_fn))
        else:
            hooks.append(blk.attn.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(inp)

    for h in hooks:
        h.remove()

    # collect only tensors with shape (B, heads, T, T) or (B, T, T)
    mats = []
    for a in attn_mats:
        if a.dim() == 4:
            mats.append(a)
        elif a.dim() == 3:
            mats.append(a.unsqueeze(1))
    if len(mats) == 0:
        raise RuntimeError("no attention maps captured. you may need to adjust hook location for your model.")

    fused = []
    for a in mats:
        if head_fusion == "mean":
            fused.append(a.mean(dim=1))  # (B,T,T)
        elif head_fusion == "max":
            fused.append(a.max(dim=1).values)
        else:
            fused.append(a.mean(dim=1))

    # rollout with residual
    device = fused[0].device
    T = fused[0].size(-1)
    result = torch.eye(T, device=device).unsqueeze(0)

    for a in fused:
        a = a + torch.eye(T, device=device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)
        result = torch.bmm(a, result)

    # CLS -> patches
    mask = result[0, 0, 1:]
    n = mask.shape[0]
    s = int(math.sqrt(n))
    mask = mask.reshape(s, s).detach().cpu().numpy()
    mask = normalize_heatmap(mask)
    # upsample to 224x224 is handled by caller via cv2 if needed; here return patch grid
    return mask
