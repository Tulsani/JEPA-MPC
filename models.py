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


class SlotAttention(nn.Module):
    """Slot Attention module for learning entity-centric representations."""
    
    def __init__(self, input_dim=256, slot_dim=256, num_slots=2, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        
        # Learnable slot initializations
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, num_slots, slot_dim) * 0.1)
        
        # Linear layers for attention
        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(input_dim, slot_dim)
        self.to_v = nn.Linear(input_dim, slot_dim)
        
        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        
        # Layer normalization
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
    def forward(self, inputs):
        """Forward pass through Slot Attention.
        
        Args:
            inputs: Input features [batch_size, num_inputs, input_dim]
            
        Returns:
            slots: Updated slot representations [batch_size, num_slots, slot_dim]
        """
        batch_size, num_inputs, input_dim = inputs.shape
        
        # Initialize slots
        slots = self.slots_mu + self.slots_sigma * torch.randn(
            batch_size, self.num_slots, self.slot_dim, device=inputs.device
        )
        
        # Multiple rounds of attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            normalized_slots = self.norm_slots(slots)
            
            # Attention
            q = self.to_q(normalized_slots)  # [batch_size, num_slots, slot_dim]
            k = self.to_k(inputs)  # [batch_size, num_inputs, slot_dim]
            v = self.to_v(inputs)  # [batch_size, num_inputs, slot_dim]
            
            # Dot product attention
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / (self.slot_dim ** 0.5)  # [batch_size, num_slots, num_inputs]
            attn = F.softmax(attn_logits, dim=-1)
            
            # Weighted mean
            updates = torch.bmm(attn, v)  # [batch_size, num_slots, slot_dim]
            
            # Update slots with GRU
            slots = slots_prev + self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(batch_size, self.num_slots, self.slot_dim)
            
            # Apply MLP to each slot
            slots = slots + self.mlp(self.norm_mlp(slots))
            
        return slots


class Encoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.repr_dim = repr_dim
        
        # Convolutional layers for processing images
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
        )
        
        # Process features into slots
        self.processor = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),
        )
        
        # Slot attention module
        self.slot_attention = SlotAttention(
            input_dim=256,
            slot_dim=repr_dim,
            num_slots=2,  # Agent and wall slots
            num_iterations=3
        )
        
        # Final projection for full representation
        self.projection = nn.Linear(repr_dim * 2, repr_dim)
        
    def forward(self, x):
        # x has shape [B, 2, H, W]
        batch_size = x.shape[0]
        
        # Extract features
        features = self.conv(x)  # [B, 256, 4, 4]
        processed = self.processor(features)  # [B, 256, 4, 4]
        
        # Reshape for slot attention: [B, 16, 256]
        features_flat = processed.flatten(2).permute(0, 2, 1)
        
        # Apply slot attention
        slots = self.slot_attention(features_flat)  # [B, 2, repr_dim]
        
        # Combine slots into a single representation
        slots_combined = slots.reshape(batch_size, -1)  # [B, 2*repr_dim]
        repr = self.projection(slots_combined)  # [B, repr_dim]
        
        return repr, slots


class GraphPredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        self.repr_dim = repr_dim
        self.hidden_dim = hidden_dim
        
        # Embedding for action
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU()
        )
        
        # Wall-to-agent interaction (graph message passing)
        self.wall_to_agent = nn.Sequential(
            nn.Linear(repr_dim * 2, hidden_dim),  # Both slots as input
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GRU for agent state prediction
        self.gru = nn.GRU(
            input_size=hidden_dim + 64,  # message + action
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Final prediction head for next agent representation
        self.prediction_head = nn.Linear(hidden_dim, repr_dim)
        
    def forward(self, slots, action):
        # slots: [B, 2, repr_dim] (first=agent, second=wall)
        # action: [B, action_dim]
        agent_slot = slots[:, 0]  # [B, repr_dim]
        wall_slot = slots[:, 1]   # [B, repr_dim]
        
        # Embed action
        action_embedding = self.action_embed(action)  # [B, 64]
        
        # Wall-to-agent message passing (how walls affect agent movement)
        combined = torch.cat([agent_slot, wall_slot], dim=1)
        message = self.wall_to_agent(combined)  # [B, hidden_dim]
        
        # Concatenate message with action embedding
        gru_input = torch.cat([message, action_embedding], dim=1)  # [B, hidden_dim+64]
        gru_input = gru_input.unsqueeze(1)  # Add sequence dim: [B, 1, hidden_dim+64]
        
        # Update agent representation with GRU
        output, _ = self.gru(gru_input)  # [B, 1, hidden_dim]
        output = output.squeeze(1)  # [B, hidden_dim]
        
        # Final prediction
        next_agent_repr = self.prediction_head(output)  # [B, repr_dim]
        
        return next_agent_repr


class JEPAModel(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        
        self.encoder = Encoder(repr_dim=repr_dim)
        self.predictor = GraphPredictor(repr_dim=repr_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.repr_dim = repr_dim
        
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
        B, T_states, C, H, W = states.shape
        T_minus_1 = actions.shape[1]  # T-1 for actions
        
        # Encode initial state
        init_state = states[:, 0]  # [B, 2, H, W]
        init_repr, init_slots = self.encoder(init_state)  # [B, repr_dim], [B, 2, repr_dim]
        
        # List to store all representations
        all_reprs = [init_repr]
        
        # Current slots (agent and wall)
        current_slots = init_slots  # [B, 2, repr_dim]
        
        # Unroll predictions recurrently
        for t in range(T_minus_1):
            current_action = actions[:, t]  # [B, 2]
            
            # Predict next agent representation
            next_agent_repr = self.predictor(current_slots, current_action)  # [B, repr_dim]
            
            # Update agent slot (first slot), keep wall slot (second slot) unchanged
            next_slots = current_slots.clone()
            next_slots[:, 0] = next_agent_repr
            
            # Combine slots to get full representation
            slots_combined = next_slots.reshape(B, -1)  # [B, 2*repr_dim]
            next_repr = self.encoder.projection(slots_combined)  # [B, repr_dim]
            
            # Add to representations list
            all_reprs.append(next_repr)
            
            # Update current slots for next step
            current_slots = next_slots
        
        # Stack all representations
        all_reprs = torch.stack(all_reprs, dim=1)  # [B, T, repr_dim]
        
        return all_reprs


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