from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, decay: float) -> None:
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1]")
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("online and target modules do not share parameter names")
    for name, target_parameter in target_parameters.items():
        target_parameter.mul_(decay).add_(online_parameters[name], alpha=1.0 - decay)

    online_buffers = dict(online.named_buffers())
    target_buffers = dict(target.named_buffers())
    for name, target_buffer in target_buffers.items():
        if name in online_buffers:
            target_buffer.copy_(online_buffers[name])
