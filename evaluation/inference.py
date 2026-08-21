import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import corner 
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.flow import RingdownPosterior
from data.dataset import RingdownDataset
from utils.parser import load_config
import argparse

def run_inference():
    '''
    Loads a trained RingdownPosterior model and performs inference on a single test waveform.
    Generates posterior samples and plots the results against the true parameters.
    '''

    # Load Config
    parser = argparse.ArgumentParser(description="Run inference on a generated waveform.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to the configuration file relative to project root.")
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    save_path = os.path.join(project_root, config['training']['save_dir'])
    print(f"Looking for model at:  {save_path}")

    # Device Configuration. Search for MPS, then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Loading model on {device} from {save_path}...")

    # Load the trained network using the architecture settings from config
    seq_len = int(config['physics']['duration'] * config['physics']['sample_rate'])

    model = RingdownPosterior(
        sequence_length=seq_len,
        model_config=config['model']
    ).float().to(device)

    try:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: '{save_path}' not found. Did train.py finish successfully?")
        return
        
    model.eval()

    # Generate a single test waveform
    print("Generating a test waveform...")
    dataset = RingdownDataset(num_samples=1)
    
    signal_tensor, true_params, snr_tensor = dataset[0]
    signal_tensor = signal_tensor.unsqueeze(0).to(device)
    true_snr = snr_tensor.item()

    print(f"Test Waveform Generated. True Optimal SNR: {true_snr:.2f}")

    # Generate Posterior Samples
    print("Running statistical inference...")
    with torch.no_grad():
        samples = model.sample(signal_tensor, num_samples=2000)

    samples = samples.squeeze(0).cpu().numpy()
    true_params = true_params.numpy()

    # Raw Cartesian Corner Plot
    labels_cart = [r"$M_f$", r"$a_f$", r"$C_{re,0}$", r"$C_{im,0}$", r"$C_{re,1}$", r"$C_{im,1}$"]
    
    fig_cart = corner.corner(
        samples, 
        truths=true_params,
        labels=labels_cart,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 11},
        color='royalblue',
        truth_color='red'
    )
    fig_cart.suptitle(f"Cartesian Parameters Corner Plot (Injected SNR: {true_snr:.1f})", fontsize=16)
    plt.show()

    # Physical Mode Resolution Corner Plot
    samples_polar = np.zeros_like(samples)
    samples_polar[:, 0] = samples[:, 0]  # Mf
    samples_polar[:, 1] = samples[:, 1]  # af
    samples_polar[:, 2] = np.sqrt(samples[:, 2]**2 + samples[:, 3]**2)  # A_0
    samples_polar[:, 3] = np.arctan2(samples[:, 3], samples[:, 2])     # phi_0
    samples_polar[:, 4] = np.sqrt(samples[:, 4]**2 + samples[:, 5]**2)  # A_1
    samples_polar[:, 5] = np.arctan2(samples[:, 5], samples[:, 4])     # phi_1

    # Transform truth values
    true_polar = np.zeros_like(true_params)
    true_polar[0] = true_params[0]
    true_polar[1] = true_params[1]
    true_polar[2] = np.sqrt(true_params[2]**2 + true_params[3]**2)
    true_polar[3] = np.arctan2(true_params[3], true_params[2])
    true_polar[4] = np.sqrt(true_params[4]**2 + true_params[5]**2)
    true_polar[5] = np.arctan2(true_params[5], true_params[4])

    labels_polar = [r"$M_f$", r"$a_f$", r"$A_0$", r"$\phi_0$", r"$A_1$", r"$\phi_1$"]
    
    fig_polar = corner.corner(
        samples_polar,
        truths=true_polar,
        labels=labels_polar,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 11},
        color='seagreen',
        truth_color='red'
    )
    fig_polar.suptitle(f"Physical Mode Resolution Corner Plot (Injected SNR: {true_snr:.1f})", fontsize=16)
    plt.show()


''' QUICK TEST '''

if __name__ == "__main__":
    run_inference()
    