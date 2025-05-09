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


class Encoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        # Agent position stream (processes channel 0)
        self.agent_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU()
        )
        
        # Wall configuration stream (processes channel 1)
        self.wall_stream = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU()
        )
        
        # Combine streams
        self.combiner = nn.Sequential(
            nn.Linear(512, repr_dim),
            nn.BatchNorm1d(repr_dim)
        )
        
        self.repr_dim = repr_dim
        
    def forward(self, x):
        # x has shape [B, 2, H, W]
        agent_feat = self.agent_stream(x[:, 0:1])  # Agent channel
        wall_feat = self.wall_stream(x[:, 1:2])    # Wall channel
        
        combined = torch.cat([agent_feat, wall_feat], dim=1)
        representation = self.combiner(combined)
        
        return representation


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        
        self.repr_dim = repr_dim
        self.hidden_dim = hidden_dim
        
        # Embedding for the action
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU()
        )
        
        # GRU for sequential prediction
        self.gru = nn.GRU(
            input_size=repr_dim + 64,  # Representation + action embedding
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,  # Added an extra layer for better temporal modeling
            dropout=0.1    # Add dropout for regularization
        )
        
        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, repr_dim),
            nn.BatchNorm1d(repr_dim)
        )
        
    def forward(self, state_repr, action, hidden=None):
        # state_repr: [B, repr_dim]
        # action: [B, action_dim]
        # hidden: Optional hidden state for stateful prediction
        
        # Embed action
        action_embed = self.action_embedding(action)  # [B, 64]
        
        # Concatenate state representation and action embedding
        combined = torch.cat([state_repr, action_embed], dim=1)  # [B, repr_dim+64]
        combined = combined.unsqueeze(1)  # Add sequence dimension: [B, 1, repr_dim+64]
        
        # Pass through GRU
        if hidden is None:
            output, new_hidden = self.gru(combined)  # [B, 1, hidden_dim]
        else:
            output, new_hidden = self.gru(combined, hidden)
        
        output = output.squeeze(1)  # [B, hidden_dim]
        
        # Final prediction
        pred = self.pred_head(output)  # [B, repr_dim]
        
        return pred, new_hidden


class JEPAModel(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        
        self.encoder = Encoder(repr_dim=repr_dim)
        self.predictor = Predictor(repr_dim=repr_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.repr_dim = repr_dim
        
    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, 2, H, W]
            During inference:
                states: [B, 1, 2, H, W]
            actions: [B, T-1, 2]
            
        Output:
            predictions: [B, T, D] where D is repr_dim
        """
        B, T, _, H, W = states.shape
        T_minus_1 = actions.shape[1]  # T-1 for actions
        
        # Encode initial state
        init_state = states[:, 0]  # [B, 2, H, W]
        current_repr = self.encoder(init_state)  # [B, repr_dim]
        
        # List to store all representations
        all_reprs = [current_repr]
        
        # Track hidden state for GRU
        hidden = None
        
        # Unroll predictions recurrently
        for t in range(T_minus_1):
            current_action = actions[:, t]  # [B, 2]
            # Predict next representation
            next_repr, hidden = self.predictor(current_repr, current_action, hidden)
            all_reprs.append(next_repr)
            current_repr = next_repr  # For next step
        
        # Stack all representations
        all_reprs = torch.stack(all_reprs, dim=1)  # [B, T, repr_dim]
        
        return all_reprs
    
    def get_target_representations(self, states):
        """
        Get target representations for all states in the sequence
        
        Args:
            states: [B, T, 2, H, W]
            
        Returns:
            target_reprs: [B, T, repr_dim]
        """
        B, T, _, H, W = states.shape
        
        target_reprs = []
        for t in range(T):
            # Encode each state individually
            state_t = states[:, t]  # [B, 2, H, W]
            repr_t = self.encoder(state_t)  # [B, repr_dim]
            target_reprs.append(repr_t)
        
        # Stack all representations
        target_reprs = torch.stack(target_reprs, dim=1)  # [B, T, repr_dim]
        
        return target_reprs
    
    def predict_multi_step(self, init_state, actions):
        """
        Predict representations for multiple steps ahead
        
        Args:
            init_state: [B, 2, H, W]
            actions: [B, T, 2]
            
        Returns:
            pred_reprs: [B, T+1, repr_dim]
        """
        B, T, _ = actions.shape
        
        # Encode initial state
        current_repr = self.encoder(init_state)  # [B, repr_dim]
        
        # List to store all representations
        all_reprs = [current_repr]
        
        # Track hidden state for GRU
        hidden = None
        
        # Unroll predictions recurrently
        for t in range(T):
            current_action = actions[:, t]  # [B, 2]
            # Predict next representation
            next_repr, hidden = self.predictor(current_repr, current_action, hidden)
            all_reprs.append(next_repr)
            current_repr = next_repr  # For next step
        
        # Stack all representations
        all_reprs = torch.stack(all_reprs, dim=1)  # [B, T+1, repr_dim]
        
        return all_reprs


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