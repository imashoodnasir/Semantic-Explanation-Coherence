from typing import Optional
import timm
import torch

def build_model(name: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    return model

def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    if not ckpt_path:
        return model
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("state_dict", state)
    new_sd = {k.replace("module.",""): v for k,v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    return model
