"""Frozen LeWM encoder: shared by latent caching and closed-loop planning so
training latents and planning latents go through identical preprocessing."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_lewm(repo: str, device: torch.device | str):
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(repo)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


def preprocess(pixels: np.ndarray | torch.Tensor, img_size: int, device: torch.device | str) -> torch.Tensor:
    """uint8 ``[B, H, W, C]`` (or ``[B, C, H, W]``) → ImageNet-normalized ``[B, 3, S, S]``."""
    images = pixels if torch.is_tensor(pixels) else torch.from_numpy(np.ascontiguousarray(pixels))
    images = images.to(device)
    if images.shape[-1] in (1, 3) and images.shape[1] not in (1, 3):
        images = images.permute(0, 3, 1, 2)
    images = images.float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    images = (images - mean) / std
    if images.shape[-1] != img_size or images.shape[-2] != img_size:
        images = F.interpolate(
            images, size=(img_size, img_size), mode="bilinear", antialias=True, align_corners=False
        )
    return images


@torch.no_grad()
def encode_images(model, pixels, img_size: int = 224, batch_size: int = 256) -> torch.Tensor:
    """uint8 images → ``[B, D]`` LeWM embeddings (projected CLS token)."""
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(pixels), batch_size):
        images = preprocess(pixels[start : start + batch_size], img_size, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            embedding = model.encode({"pixels": images[:, None]})["emb"][:, 0]
        outputs.append(embedding.float())
    return torch.cat(outputs)
