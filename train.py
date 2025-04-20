import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from models import JEPAModel  # <-- Import your new model here


def vicreg_loss(pred_repr, target_repr, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
    sim_loss = F.mse_loss(pred_repr, target_repr)

    std_pred = torch.sqrt(pred_repr.var(dim=0) + 1e-4)
    std_target = torch.sqrt(target_repr.var(dim=0) + 1e-4)

    var_loss_pred = torch.mean(F.relu(1 - std_pred))
    var_loss_target = torch.mean(F.relu(1 - std_target))
    var_loss = var_loss_pred + var_loss_target

    pred_repr = pred_repr - pred_repr.mean(dim=0)
    target_repr = target_repr - target_repr.mean(dim=0)

    B = pred_repr.shape[0]
    cov_pred = (pred_repr.T @ pred_repr) / (B - 1)
    cov_target = (target_repr.T @ target_repr) / (B - 1)

    cov_pred = cov_pred - torch.diag(torch.diag(cov_pred))
    cov_target = cov_target - torch.diag(torch.diag(cov_target))

    cov_loss = (cov_pred**2).sum() / pred_repr.shape[1] + (cov_target**2).sum() / target_repr.shape[1]

    total_loss = sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss

    return total_loss, {
        'sim_loss': sim_loss.item(),
        'var_loss': var_loss.item(),
        'cov_loss': cov_loss.item(),
        'total_loss': total_loss.item()
    }


# ---------------------------- #
#        Training Setup        #
# ---------------------------- #

# Hyperparameters
repr_dim = 256
shared_dim = 128
batch_size = 64
learning_rate = 0.001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_training_data(data_path="/scratch/DL25SP/train"):
    states = np.load(f"{data_path}/states.npy")
    actions = np.load(f"{data_path}/actions.npy")
    print(f"Loaded states with shape {states.shape}")
    print(f"Loaded actions with shape {actions.shape}")
    return states, actions


def create_dataloader(states, actions, batch_size=64):
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.FloatTensor(actions)
    dataset = TensorDataset(states_tensor, actions_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def train_jepa(model, dataloader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, (states, actions) in enumerate(pbar):
                states = states.to(device)     # [B, T, 2, H, W]
                actions = actions.to(device)   # [B, T-1, 2]

                optimizer.zero_grad()

                # Forward through JEPA model
                pred_reprs = model(states, actions)  # [B, T, repr_dim]
                B, T, D = pred_reprs.shape
                H, W = states.shape[-2], states.shape[-1]

                # Compute target representations using agent_proj only
                flat = states.view(B*T, 2, H, W)
                shared = model.shared_encoder(flat)
                agent_feat = F.relu(model.agent_proj(shared))
                target_reprs = agent_feat.view(B, T, D)

                # Compute VicReg loss over time (excluding t=0)
                P = pred_reprs[:, 1:].reshape(-1, D)
                Q = target_reprs[:, 1:].reshape(-1, D)
                loss, info = vicreg_loss(P, Q)

                loss.backward()
                optimizer.step()

                pbar.set_postfix({
                    "loss": info["total_loss"],
                    "sim" : info["sim_loss"],
                    "var" : info["var_loss"],
                    "cov" : info["cov_loss"]
                })

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': info["total_loss"],
        }, f'jepa_checkpoint_epoch_{epoch+1}.pt')

    # Save final model
    torch.save(model.state_dict(), 'jepa_final_model.pt')
    return model


def main():
    print("Loading data...")
    states, actions = load_training_data()

    print("Creating dataloader...")
    dataloader = create_dataloader(states, actions, batch_size=batch_size)

    print("Initializing model...")
    model = JEPAModel(repr_dim=repr_dim, shared_dim=shared_dim).to(device)
    print("Model parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    trained_model = train_jepa(model, dataloader, optimizer, epochs)

    print("Training complete. Model saved as 'jepa_final_model.pt'")
    return trained_model


if __name__ == "__main__":
    main()
