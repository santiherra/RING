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

    # Load configuration
    parser = argparse.ArgumentParser(description="Run inference on a generated waveform.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to the configuration file relative to project root.")
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    save_path = os.path.join(project_root, config['training']['save_dir'])
    print(f"Looking for model at:  {save_path}")

    # Device configuration. Search for MPS, then CUDA, else CPU.
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

    print(f"Test waveform generated. True optimal SNR: {true_snr:.2f}")

    # Generate posterior samples
    print("Running statistical inference...")
    num_samples = config.get('inference', {}).get('num_samples', 2000)
    
    with torch.no_grad():
        samples = model.sample(signal_tensor, num_samples=num_samples)

    samples = samples.squeeze(0).cpu().numpy()
    true_params = true_params.numpy()

    # Cartesian corner plot
    labels_cart = [r"$M_f$", r"$a_f$", 
                   r"$x_{+,0}$", r"$y_{+,0}$", r"$x_{\times,0}$", r"$y_{\times,0}$",
                   r"$x_{+,1}$", r"$y_{+,1}$", r"$x_{\times,1}$", r"$y_{\times,1}$"]
    
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
    fig_cart.suptitle(f"Corner plot, cartesian parameters (Injected SNR: {true_snr:.1f})", fontsize=16)
    plt.show()

    # Polar corner plot
    def cart_to_polar(x_p, y_p, x_c, y_c):
        term1 = np.sqrt((x_p + y_c)**2 + (x_c - y_p)**2)
        term2 = np.sqrt((x_p - y_c)**2 + (x_c + y_p)**2)
        A = 0.5 * (term1 + term2)
        denom = term1 + term2
        eps = np.where(denom > 0, (term1 - term2) / denom, 0.0)
        theta = -0.5 * (np.arctan2(-x_c + y_p, y_c + x_p) + np.arctan2(-x_c - y_p, -y_c + x_p))
        phi = 0.5 * (np.arctan2(-x_c + y_p, y_c + x_p) - np.arctan2(-x_c - y_p, -y_c + x_p))
        return A, eps, theta, phi

    samples_polar = np.zeros_like(samples)
    samples_polar[:, 0] = samples[:, 0]  # Mf
    samples_polar[:, 1] = samples[:, 1]  # af
    samples_polar[:, 2:6] = np.column_stack(cart_to_polar(samples[:,2], samples[:,3], samples[:,4], samples[:,5])) 
    samples_polar[:, 6:10] = np.column_stack(cart_to_polar(samples[:,6], samples[:,7], samples[:,8], samples[:,9]))

    # Transform values
    true_polar = np.zeros_like(true_params)
    true_polar[0], true_polar[1] = true_params[0], true_params[1] # Mf, af
    true_polar[2:6] = cart_to_polar(true_params[2], true_params[3], true_params[4], true_params[5]) # A_0, eps_0, theta_0, phi_0
    true_polar[6:10] = cart_to_polar(true_params[6], true_params[7], true_params[8], true_params[9]) # A_1, eps_1, theta_1, phi_1

    labels_polar = [r"$M_f$", r"$a_f$", 
                    r"$A_0$", r"$\epsilon_0$", r"$\theta_0$", r"$\phi_0$",
                    r"$A_1$", r"$\epsilon_1$", r"$\theta_1$", r"$\phi_1$"]
    
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
    fig_polar.suptitle(f"Corner plot, polar parameters (Injected SNR: {true_snr:.1f})", fontsize=16)
    plt.show()


''' QUICK TEST '''

if __name__ == "__main__":
    run_inference()
    