from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class SharedConvEncoder(nn.Module):
    """Single small ConvNet that ingests both channels at once."""
    def __init__(self, shared_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,  32, 3, stride=2, padding=1),  # 64→32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32→16
            nn.ReLU(True),
            nn.Conv2d(64, shared_dim, 3, stride=2, padding=1),  # 16→8
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # → [B, shared_dim, 1, 1]
            nn.Flatten(),             # → [B, shared_dim]
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 2, H, W]
        return self.net(x)         # [B, shared_dim]


class JEPAModel(nn.Module):
    """
    JEPA model where 'walls' are encoded once per trajectory as static context.
    Residual predictor only moves the 'agent' embedding.
    """
    def __init__(self, repr_dim=128, action_dim=2, shared_dim=128):
        super().__init__()
        self.shared_encoder = SharedConvEncoder(shared_dim)
        self.agent_proj = nn.Linear(shared_dim, repr_dim)
        self.wall_proj  = nn.Linear(shared_dim, repr_dim)
        # predictor: [agent_prev, wall_const, action] → delta
        self.transition = build_mlp([repr_dim*2 + action_dim, repr_dim, repr_dim])
        self.repr_dim = repr_dim

    def forward(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            states:  [B, T, 2, H, W]
            actions: [B, T-1, 2]
        Returns:
            reps:    [B, T, repr_dim]
        """
        B, T, _, H, W = states.shape

        # 1) encode both channels of first frame once
        first = states[:, 0]                            # [B, 2, H, W]
        shared_feat = self.shared_encoder(first)        # [B, shared_dim]
        agent0 = F.relu(self.agent_proj(shared_feat))   # [B, repr_dim]
        wall0  = F.relu(self.wall_proj(shared_feat))    # [B, repr_dim]

        # 2) unroll residual predictor
        seq = [agent0]
        for t in range(actions.size(1)):
            a_t = actions[:, t]                         # [B, 2]
            inp = torch.cat([ seq[-1], wall0, a_t ], dim=-1)  # [B, 2*repr_dim + 2]
            delta = self.transition(inp)                # [B, repr_dim]
            next_repr = seq[-1] + delta                 # residual update
            seq.append(next_repr)

        # 3) stack → [B, T, repr_dim]
        return torch.stack(seq, dim=1)

class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
