import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from models import JEPAModel
import torch.nn.functional as F


def vicreg_loss(pred_repr, target_repr, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
    """
    VicReg loss with three terms:
    1. Invariance loss - making representations similar
    2. Variance loss - ensuring high variance in each dimension
    3. Covariance loss - decorrelating dimensions
    
    Args:
        pred_repr: Predicted representations [B, D]
        target_repr: Target representations [B, D]
        sim_coef: Weight for similarity loss
        var_coef: Weight for variance loss
        cov_coef: Weight for covariance loss
    """
    # 1. Similarity/Invariance loss (MSE)
    sim_loss = F.mse_loss(pred_repr, target_repr)
    
    # 2. Variance loss
    std_pred = torch.sqrt(pred_repr.var(dim=0) + 1e-4)
    std_target = torch.sqrt(target_repr.var(dim=0) + 1e-4)
    
    var_loss_pred = torch.mean(F.relu(1 - std_pred))
    var_loss_target = torch.mean(F.relu(1 - std_target))
    var_loss = var_loss_pred + var_loss_target
    
    # 3. Covariance loss
    pred_repr = pred_repr - pred_repr.mean(dim=0)
    target_repr = target_repr - target_repr.mean(dim=0)
    
    B = pred_repr.shape[0]
    cov_pred = (pred_repr.T @ pred_repr) / (B - 1)
    cov_target = (target_repr.T @ target_repr) / (B - 1)
    
    # Remove diagonal (variances) and focus on off-diagonal (covariances)
    cov_pred = cov_pred - torch.diag(torch.diag(cov_pred))
    cov_target = cov_target - torch.diag(torch.diag(cov_target))
    
    cov_loss = (cov_pred**2).sum() / pred_repr.shape[1] + (cov_target**2).sum() / target_repr.shape[1]
    
    # Combine losses
    total_loss = sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss
    
    return total_loss, {
        'sim_loss': sim_loss.item(),
        'var_loss': var_loss.item(),
        'cov_loss': cov_loss.item(),
        'total_loss': total_loss.item()
    }

# Hyperparameters
repr_dim = 256
hidden_dim = 256
batch_size = 64
learning_rate = 0.001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
def load_training_data(data_path="/scratch/DL25SP/train"):
    states = np.load(f"{data_path}/states.npy",mmap_mode="r")
    actions = np.load(f"{data_path}/actions.npy")
    
    print(f"Loaded states with shape {states.shape}")
    print(f"Loaded actions with shape {actions.shape}")
    
    return states, actions

def create_dataloader(states, actions, batch_size=64):
    # Convert to PyTorch tensors
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.FloatTensor(actions)
    
    # Create dataset
    dataset = TensorDataset(states_tensor, actions_tensor)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    return dataloader

# Training function
def train_jepa(model, dataloader, optimizer, epochs):
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        sim_losses = 0
        var_losses = 0
        cov_losses = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, (states, actions) in enumerate(pbar):
                states = states.to(device)
                actions = actions.to(device)
                
                optimizer.zero_grad()
                
                # Teacher forcing approach
                B, T, C, H, W = states.shape
                
                # Encode all states with the encoder
                target_reprs = []
                for t in range(T):
                    target_reprs.append(model.encoder(states[:, t]))
                
                target_reprs = torch.stack(target_reprs, dim=1)  # [B, T, repr_dim]
                
                # Predict representations recurrently
                init_state = states[:, 0]
                current_repr = model.encoder(init_state)
                
                pred_reprs = [current_repr]
                for t in range(T-1):
                    current_action = actions[:, t]
                    next_repr = model.predictor(current_repr, current_action)
                    pred_reprs.append(next_repr)
                    current_repr = next_repr  # For next step
                
                pred_reprs = torch.stack(pred_reprs, dim=1)  # [B, T, repr_dim]
                
                # Calculate loss for each timestep except the first (which is the same)
                losses = []
                loss_info = {'sim_loss': 0, 'var_loss': 0, 'cov_loss': 0, 'total_loss': 0}
                
                for t in range(1, T):
                    loss, info = vicreg_loss(
                        pred_reprs[:, t], 
                        target_reprs[:, t]
                    )
                    losses.append(loss)
                    for k, v in info.items():
                        loss_info[k] += v
                
                # Average the losses    
                total_loss = sum(losses) / len(losses)
                for k in loss_info:
                    loss_info[k] /= len(losses)
                
                # Backpropagate and update
                total_loss.backward()
                optimizer.step()
                
                # Log
                epoch_loss += total_loss.item()
                sim_losses += loss_info['sim_loss']
                var_losses += loss_info['var_loss']
                cov_losses += loss_info['cov_loss']
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': epoch_loss / (batch_idx + 1),
                    'sim': sim_losses / (batch_idx + 1),
                    'var': var_losses / (batch_idx + 1),
                    'cov': cov_losses / (batch_idx + 1)
                })
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(dataloader),
            }, f'jepa_checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), 'jepa_final_model.pt')
    return model

# Main training function
def main():
    print("Loading data...")
    states, actions = load_training_data()
    
    print("Creating dataloader...")
    dataloader = create_dataloader(states, actions, batch_size=batch_size)
    
    print("Initializing model...")
    model = JEPAModel(repr_dim=repr_dim, hidden_dim=hidden_dim).to(device)
    
    print("Model parameter count:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    trained_model = train_jepa(model, dataloader, optimizer, epochs)
    
    print("Training complete. Model saved as 'jepa_final_model.pt'")
    return trained_model

if __name__ == "__main__":
    main()