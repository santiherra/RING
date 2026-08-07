import os
import sys
import time
import torch
from torch.optim import Adam

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataloader import get_dataloaders
from models.flow import RingdownPosterior
from utils.parser import load_config

def train():
    '''
    Trains the RingdownPosterior model using the specified configuration and 
    saves the best model based on validation loss.
    '''

    # Load Configuration
    config = load_config("configs/config.yaml")
    
    # Device Configuration. Search for MPS (Apple Silicon), then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # Read Hyperparameters from Config
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_train = config['training']['num_train_samples']
    num_val = config['training']['num_val_samples']
    
    # Initialize DataLoaders 
    print("Generating physics dataset... (This may take a moment)")
    train_loader, val_loader = get_dataloaders(
        num_train=num_train, 
        num_val=num_val, 
        batch_size=batch_size
    )
    
    # Initialize model using architecture settings from config
    model = RingdownPosterior(
        context_dim=config['model']['context_dim'],
        hidden_features=config['model']['hidden_features'],
        num_transforms=config['model']['num_transforms']
    ).float().to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print("Starting Training Loop...\n")
    print("-" * 30)

    # Resolve absolute path to guarantee saving inside the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(project_root, config['training']['save_dir'])
    
    # Ensure the parent folder exists (exist_ok=True reuses the existing folder)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_val_loss = float('inf')
    
    # The Epoch Loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # TRAINING PHASE
        model.train()
        train_loss = 0.0

        # For each batch, perform a forward pass, compute the loss, backpropagate, and update the model parameters.
        for batch_idx, (signals, targets) in enumerate(train_loader):
            signals = signals.float().to(device)
            targets = targets.float().to(device)
            
            optimizer.zero_grad()
            loss = model(signals, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION PHASE
        model.eval()
        val_loss = 0.0

        # Disable gradient computation for validation to save memory and computations
        with torch.no_grad():
            for signals, targets in val_loader:
                signals = signals.float().to(device)
                targets = targets.float().to(device)
                
                loss = model(signals, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}] | Time: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save / overwrite the model directly inside saved_models/
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            
    print("-" * 30)
    print(f"Training Complete! Model saved directly to: '{save_path}'")


''' QUICK TEST '''

if __name__ == "__main__":
    train()
    