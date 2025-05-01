import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from models import JEPAModel
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


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


def multi_step_vicreg_loss(pred_reprs, target_reprs, unroll_steps=[1, 3, 5], weights=None):
    """
    Apply VicReg loss at multiple time horizons to enforce temporal coherence
    
    Args:
        pred_reprs: Predicted representations [B, T, D]
        target_reprs: Target representations [B, T, D]
        unroll_steps: List of time horizons to apply loss at
        weights: Optional weights for each horizon loss (default: equal weights)
    """
    B, T, D = pred_reprs.shape
    
    if weights is None:
        # Equal weighting by default, but decreasing with horizon length could also work
        weights = [1.0] * len(unroll_steps)
    
    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    losses = []
    loss_infos = []
    
    for i, step in enumerate(unroll_steps):
        if step >= T:
            continue
            
        # For each horizon length, calculate losses for all possible starting points
        horizon_losses = []
        horizon_infos = []
        
        # Apply loss for each valid starting position
        for start in range(T - step):
            end = start + step
            loss, info = vicreg_loss(pred_reprs[:, end], target_reprs[:, end])
            horizon_losses.append(loss)
            horizon_infos.append(info)
        
        # Average losses for this horizon length
        if horizon_losses:
            avg_loss = sum(horizon_losses) / len(horizon_losses)
            losses.append(weights[i] * avg_loss)
            
            # Average loss info
            avg_info = {}
            for k in horizon_infos[0].keys():
                avg_info[k] = sum(info[k] for info in horizon_infos) / len(horizon_infos)
            loss_infos.append(avg_info)
    
    # Combine all horizon losses
    total_loss = sum(losses)
    
    # Combine loss info
    combined_info = {}
    for k in loss_infos[0].keys():
        combined_info[k] = sum(info[k] for info in loss_infos) / len(loss_infos)
    combined_info['total_loss'] = total_loss.item()
    
    return total_loss, combined_info


# Data augmentation functions
def random_flip(states, actions):
    """Randomly flip the x-axis (horizontal flip)"""
    if torch.rand(1) < 0.5:
        # Flip states horizontally
        states = torch.flip(states, dims=[3])  # Flip width dimension
        # Flip x component of actions
        actions[..., 0] = -actions[..., 0]
    return states, actions

def random_crop_and_resize(states, size=64):
    """Randomly crop and resize back to original size"""
    B, T, C, H, W = states.shape
    
    # Flatten batch and time dimensions for processing
    states_flat = states.view(-1, C, H, W)
    
    # Random crop size (between 80% and 100% of original)
    crop_size = int(torch.rand(1) * 0.2 * size + 0.8 * size)
    
    # Random crop
    i = torch.randint(0, H - crop_size + 1, (1,))
    j = torch.randint(0, W - crop_size + 1, (1,))
    states_cropped = states_flat[:, :, i:i+crop_size, j:j+crop_size]
    
    # Resize back to original
    states_resized = F.interpolate(states_cropped, size=(H, W), mode='bilinear', align_corners=False)
    
    # Reshape back to original dimensions
    states = states_resized.view(B, T, C, H, W)
    
    return states

def apply_augmentations(states, actions):
    """Apply a series of augmentations to the data"""
    # Flipping (affects both states and actions)
    states, actions = random_flip(states, actions)
    
    # Cropping (only affects states)
    if torch.rand(1) < 0.5:  # 50% chance of applying crop
        states = random_crop_and_resize(states)
    
    return states, actions


# Hyperparameters
repr_dim = 256
hidden_dim = 256
batch_size = 64
learning_rate = 0.001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define multi-step horizons for prediction
unroll_horizons = [1, 3, 5]  # Predict 1, 3, and 5 steps ahead
horizon_weights = [0.5, 0.3, 0.2]  # Higher weight to shorter horizons

# Learning rate scheduler parameters
lr_warmup_epochs = 5
lr_decay_epochs = 50

# Create output directory for checkpoints and logs
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)


# Load data
def load_training_data(data_path="/scratch/DL25SP/train"):
    states = np.load(f"{data_path}/states.npy", mmap_mode="r")
    actions = np.load(f"{data_path}/actions.npy")
    
    print(f"Loaded states with shape {states.shape}")
    print(f"Loaded actions with shape {actions.shape}")
    
    return states, actions


def create_dataloader(states, actions, batch_size=64):
    # Create a sampler for efficient loading with memory-mapped arrays
    num_trajectories = states.shape[0]
    
    # Convert to PyTorch tensors
    # For memory-mapped states, we'll load batches on-the-fly
    actions_tensor = torch.FloatTensor(actions)
    
    # Create dataset class that loads states on demand
    class TrajectoryDataset(torch.utils.data.Dataset):
        def __init__(self, states, actions):
            self.states = states
            self.actions = actions
        
        def __len__(self):
            return len(self.actions)
        
        def __getitem__(self, idx):
            # Load state on demand from memory-mapped array
            state = torch.FloatTensor(self.states[idx])
            action = self.actions[idx]
            return state, action
    
    dataset = TrajectoryDataset(states, actions_tensor)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster data transfer to GPU
    )
    
    return dataloader


# Learning rate scheduler
def get_lr_scheduler(optimizer, warmup_epochs, decay_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            decay_ratio = (epoch - warmup_epochs) / decay_epochs
            decay_ratio = min(decay_ratio, 1.0)  # Cap at 1.0
            return 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Training function
def train_jepa(model, dataloader, optimizer, epochs):
    model.train()
    
    # Initialize learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, lr_warmup_epochs, lr_decay_epochs, epochs)
    
    # Lists to track metrics
    epoch_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        sim_losses = 0
        var_losses = 0
        cov_losses = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, (states, actions) in enumerate(pbar):
                states = states.to(device)
                actions = actions.to(device)
                
                # Apply data augmentations
                states, actions = apply_augmentations(states, actions)
                
                optimizer.zero_grad()
                
                # Get target representations for all states
                target_reprs = model.get_target_representations(states)
                
                # Predict representations recurrently
                pred_reprs = model.predict_multi_step(states[:, 0], actions)
                
                # Calculate multi-step losses
                total_loss, loss_info = multi_step_vicreg_loss(
                    pred_reprs, 
                    target_reprs,
                    unroll_steps=unroll_horizons,
                    weights=horizon_weights
                )
                
                # Backpropagate and update
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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
                    'cov': cov_losses / (batch_idx + 1),
                    'lr': optimizer.param_groups[0]['lr']
                })
        
        # Update learning rate
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = f'checkpoints/jepa_checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Plot and save loss curve
        if epoch > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), epoch_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.grid(True)
            plt.savefig('logs/loss_curve.png')
            plt.close()
    
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
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    print("Starting training...")
    trained_model = train_jepa(model, dataloader, optimizer, epochs)
    
    print("Training complete. Model saved as 'jepa_final_model.pt'")
    return trained_model


if __name__ == "__main__":
    main()