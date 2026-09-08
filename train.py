import os
import sys
import time
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataloader import get_dataloaders
from models.flow import RingdownPosterior
from utils.parser import load_config
import argparse

def train():
    '''
    Trains the RingdownPosterior model using the specified configuration and 
    saves the best model based on validation loss.
    '''

    # Load configuration file
    parser = argparse.ArgumentParser(description="Train the Ringdown Normalizing Flow.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to the configuration file relative to the project root.")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, args.config)
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Device configuration. Search for MPS, then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # Read hyperparameters from config file
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    num_train = config['training']['num_train_samples']
    num_val = config['training']['num_val_samples']
    
    # Initialize DataLoaders 
    print("Generating signal dataset...")
    train_loader, val_loader = get_dataloaders(
        num_train=num_train, 
        num_val=num_val, 
        batch_size=batch_size,
        config_path=config_path
    )
    
    # Initialize model
    seq_len = int(config['physics']['duration'] * config['physics']['sample_rate'])

    model = RingdownPosterior(
        sequence_length=seq_len,
        model_config=config['model']
    ).float().to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)

    print("Starting training process...\n")

    # Save the model
    project_root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(project_root, config['training']['save_dir'])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_val_loss = float('inf')

    history_train_loss = []
    history_val_loss = []
    tracked_snrs = []
    
    # The epoch loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # TRAINING PHASE
        model.train()
        train_loss = 0.0

        # For each batch perform a forward pass, compute the loss, backpropagate, and update the model parameters
        for batch_idx, (signals, targets, snrs) in enumerate(train_loader):
            signals = signals.float().to(device)
            targets = targets.float().to(device)

            if epoch == 0 and config.get('diagnostics', {}).get('compute_snr', True):
                tracked_snrs.extend(snrs.flatten().tolist())
            
            optimizer.zero_grad()
            loss = model(signals, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        
        # VALIDATION PHASE
        model.eval()
        val_loss = 0.0

        # Disable gradient computation for validation
        with torch.no_grad():
            for signals, targets, snrs in val_loader:
                signals = signals.float().to(device)
                targets = targets.float().to(device)
                
                loss = model(signals, targets)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        history_val_loss.append(avg_val_loss)

        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{epochs}].  Time: {epoch_time:.1f}s.  "
              f"Train Loss: {avg_train_loss:.4f}.  Val Loss: {avg_val_loss:.4f}")
        
        # Save the model inside saved_models/
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            
    print(f"Training Complete. Model saved directly to: '{save_path}'")

    # Generate Diagnostics if 'true' in config.yaml
    diagnostics_dir = os.path.join(project_root, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    if config.get('diagnostics', {}).get('save_history_data', False):
        print("Saving training data...")
        
        losses_data = np.column_stack((history_train_loss, history_val_loss))
        np.savetxt(
            os.path.join(diagnostics_dir, "losses.txt"), 
            losses_data, 
            header="Train_Loss Val_Loss", 
            comments=''
        )
        
        if config.get('diagnostics', {}).get('compute_snr', True):
            np.savetxt(
                os.path.join(diagnostics_dir, "snr_data.txt"), 
                tracked_snrs, 
                header="Optimal_SNR", 
                comments=''
            )

    if config.get('diagnostics', {}).get('plot_loss_curve', False):
        print("Generating loss curve plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), history_train_loss, label='Training Loss', color='royalblue', linewidth=2)
        plt.plot(range(1, epochs + 1), history_val_loss, label='Validation Loss', color='darkorange', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylim(0, 20)
        plt.title('Dataset Learning Curves')
        plt.legend()
        
        curve_path = os.path.join(diagnostics_dir, "loss_curve.png")
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close() # Free up memory
        print(f"Loss curve saved to: '{curve_path}'")

    if config.get('diagnostics', {}).get('plot_snr_distribution', False):
        plt.figure(figsize=(10, 6))
        plt.hist(tracked_snrs, bins=40, color='mediumseagreen', edgecolor='black', alpha=0.7)
        plt.xlabel('SNR')
        plt.title('Training Dataset SNR Distribution')
        snr_path = os.path.join(diagnostics_dir, "snr_distribution.png")
        plt.savefig(snr_path, dpi=300, bbox_inches='tight')
        print(f"SNR distribution saved to: '{snr_path}'")


''' QUICK TEST '''

if __name__ == "__main__":
    train()
    