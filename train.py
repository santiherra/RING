import os
import sys
import time
import numpy as np
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
import bilby

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataloader import get_dataloaders
from models.flow import RingdownPosterior
from physics.detector import get_detector_responses
from utils.parser import load_config
from physics.transforms import get_gpu_noise_and_whiten
import argparse

def train():
    '''
    Trains the RingdownPosterior model using the configuration and saves the best model based on validation loss.
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # Read hyperparameters from config file
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    num_train = config['training']['num_train_samples']
    num_val = config['training']['num_val_samples']

    lr_stage_1 = config['training'].get('lr_stage_1', 0.0005)
    lr_stage_2 = config['training'].get('lr_stage_2', 0.00005)
    epoch_switch = config['training'].get('epoch_switch', 100)
    weight_decay = config['training'].get('weight_decay', 0.01)

    sample_rate = config['physics']['sample_rate']
    duration = config['physics']['duration']
    scale = float(config['physics']['scale'])
    seq_len = int(duration * sample_rate)

    detector_names = config['event'].get('detectors', ['H1', 'L1'])
    config['model']['embedding_net']['in_channels'] = len(detector_names)
    print(f"Detectors in network: {detector_names} \n")

    responses = get_detector_responses(
        detector_names=detector_names,
        ra=config['event']['fixed_ra'],
        dec=config['event']['fixed_dec'],
        psi=config['event']['fixed_psi'],
        gps_time=config['event']['fixed_gps'],
        sample_rate=sample_rate,
        duration=duration
    )

    F_plus = torch.tensor(responses['F_plus'], dtype=torch.float32, device=device)
    F_cross = torch.tensor(responses['F_cross'], dtype=torch.float32, device=device)
    time_delays = torch.tensor(responses['time_delay'], dtype=torch.float32, device=device)
    asd_tensors = torch.tensor(responses['asd'], dtype=torch.float32, device=device)
    
    # Initialize data loaders 
    print("Loading signal datasets...")
    train_loader, val_loader = get_dataloaders(
        num_train=num_train, 
        num_val=num_val, 
        batch_size=batch_size,
        config_path=config_path
    )
    
    # Initialize model
    model = RingdownPosterior(
        sequence_length=seq_len,
        model_config=config['model']
    ).float().to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr_stage_1, weight_decay=weight_decay)

    print("Starting training process...\n")

    # Save the model
    save_path = os.path.join(project_root, config['training']['save_dir'])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_val_loss = float('inf')
    start_epoch = 0
    history_train_loss = []
    history_val_loss = []
    tracked_snr = []

    checkpoint_path = os.path.join(os.path.dirname(save_path), "checkpoint.pth")
    diagnostics_dir = os.path.join(project_root, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Checkpoint resume
    if os.path.exists(checkpoint_path):
        print(f"Found '{checkpoint_path}'. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        history_train_loss = checkpoint['history_train_loss']
        history_val_loss = checkpoint['history_val_loss']
        tracked_snr = checkpoint.get('tracked_snr', checkpoint.get('tracked_snr', []))
        print(f"Resuming from epoch {start_epoch + 1}...\n")

    # Time tracking
    global_start_time = time.time()
    max_runtime = config['training'].get('max_runtime_minutes', float('inf')) * 60

    # The epoch loop
    try:
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            
            # TRAINING PHASE
            current_lr = lr_stage_1 if epoch < epoch_switch else lr_stage_2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            model.train()
            train_loss = 0.0

            for hp_b, hx_b, targets in train_loader:
                hp_b = hp_b.float().to(device)
                hx_b = hx_b.float().to(device)
                targets = targets.float().to(device)
                
                signals, snr = get_gpu_noise_and_whiten(
                    hp_b, hx_b, F_plus, F_cross, time_delays, asd_tensors, scale, sample_rate, duration, device
                )

                if epoch == 0 and config.get('diagnostics', {}).get('compute_snr', True):
                    tracked_snr.extend(snr.cpu().flatten().tolist())

                optimizer.zero_grad()
                loss = model(signals, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history_train_loss.append(avg_train_loss)
            
            model.eval()
            val_loss = 0.0
            
            # VALIDATION PHASE
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for hp_b, hx_b, targets in val_loader:
                    hp_b = hp_b.float().to(device)
                    hx_b = hx_b.float().to(device)
                    targets = targets.float().to(device)
                    
                    signals, _ = get_gpu_noise_and_whiten(
                        hp_b, hx_b, F_plus, F_cross, time_delays, asd_tensors, scale, sample_rate, duration, device
                    )
                    
                    loss = model(signals, targets)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            history_val_loss.append(avg_val_loss)

            epoch_time = time.time() - start_time
            
            print(f"Epoch [{epoch+1}/{epochs}].  Time: {epoch_time:.1f}s.  "
                  f"Train loss: {avg_train_loss:.4f}.  Val loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                
            # SAVE CHECKPOINT
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history_train_loss': history_train_loss,
                'history_val_loss': history_val_loss,
                'tracked_snr': tracked_snr
            }
            torch.save(checkpoint, checkpoint_path)

            if config.get('diagnostics', {}).get('save_history_data', False):
                losses_data = np.column_stack((history_train_loss, history_val_loss))
                np.savetxt(os.path.join(diagnostics_dir, "losses.txt"), losses_data, header="Train_Loss Val_Loss", comments='')
                if config.get('diagnostics', {}).get('compute_snr', True) and len(tracked_snr) > 0:
                    np.savetxt(os.path.join(diagnostics_dir, "snr_data.txt"), tracked_snr, header="Optimal_SNR", comments='')

            # Time limit check
            if time.time() - global_start_time > max_runtime:
                print(f"\nWARNING: Time limit of {max_runtime/60:.1f} minutes reached. Saving and exiting.")
                break

    except KeyboardInterrupt:
        print("\nWARNING: Training interrupted. Checkpoint and data saved safely.")

    print(f"Training complete. Model stored at: '{save_path}'")

    # Generate diagnostics if 'true' in config.yaml
    diagnostics_dir = os.path.join(project_root, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    if config.get('diagnostics', {}).get('plot_loss_curve', False):
        print("Generating loss curve plot...")
        actual_epochs = len(history_train_loss)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, actual_epochs + 1), history_train_loss, label='Training Loss', color='royalblue', linewidth=2)
        plt.plot(range(1, actual_epochs + 1), history_val_loss, label='Validation Loss', color='darkorange', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylim(0, 20)
        plt.title('Dataset Learning Curves')
        plt.legend()
        
        curve_path = os.path.join(diagnostics_dir, "loss_curve.png")
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to: '{curve_path}'")

    if config.get('diagnostics', {}).get('plot_snr_distribution', False) and len(tracked_snr) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(tracked_snr, bins=40, color='mediumseagreen', edgecolor='black', alpha=0.7)
        plt.xlabel('SNR')
        plt.title('Training Dataset SNR Distribution')
        plt.savefig(os.path.join(diagnostics_dir, "snr_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()


''' QUICK TEST '''

if __name__ == "__main__":
    train()
    