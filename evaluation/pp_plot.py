import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)

if project_root not in sys.path:
    sys.path.insert(0, project_root)    

from models.flow import RingdownPosterior
from data.dataset import RingdownDataset
from utils.parser import load_config
import argparse

def run_pp_analysis():
    '''
    Run P-P plot analysis for the RingdownPosterior model.
    '''
    # Load Configuration
    parser = argparse.ArgumentParser(description="Run P-P Plot Calibration.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to the configuration file relative to project root.")
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    pp_config = config.get('diagnostics', {}).get('pp_plot', {})
    num_test_events = pp_config.get('num_test_events', 100)
    num_samples_per_event = pp_config.get('num_samples_per_event', 2000)
    
    save_path = os.path.join(project_root, config['training']['save_dir'])
    diagnostics_dir = os.path.join(project_root, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Device Configuration. Search for MPS, then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running P-P Plot Analysis on {device} across {num_test_events} test events...")

    # Load Model
    model = RingdownPosterior(
        in_channels=config['model']['in_channels'],
        param_dim=config['model']['param_dim'],
        context_dim=config['model']['context_dim'],
        hidden_features=config['model']['hidden_features'],
        num_transforms=config['model']['num_transforms']
    ).float().to(device)
    
    try:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: '{save_path}' not found. Please run train.py first.")
        return
        
    model.eval()

    # Generate Test Dataset
    print(f"Generating {num_test_events} test waveforms...")
    test_dataset = RingdownDataset(num_samples=num_test_events)
    
    # Target parameter names
    param_names = [r"$M_f$", r"$a_f$", r"$C_{re,0}$", r"$C_{im,0}$", r"$C_{re,1}$", r"$C_{im,1}$"]
    num_params = len(param_names)
    
    # Matrix to store percentile ranks: shape [num_test_events, num_params]
    percentiles = np.zeros((num_test_events, num_params))

    # Loop over all test events
    print("Evaluating posteriors...")
    for i in range(num_test_events):
        signal_tensor, true_params, _ = test_dataset[i]
        signal_tensor = signal_tensor.unsqueeze(0).to(device)
        true_params = true_params.numpy()

        with torch.no_grad():
            # Sample posterior from normalizing flow
            samples = model.sample(signal_tensor, num_samples=num_samples_per_event)
            samples = samples.squeeze(0).cpu().numpy()

        # Calculate percentile rank for each parameter
        for j in range(num_params):
            rank = np.sum(samples[:, j] <= true_params[j]) / num_samples_per_event
            percentiles[i, j] = rank

        if (i + 1) % 20 == 0 or (i + 1) == num_test_events:
            print(f"Processed [{i + 1}/{num_test_events}] events.")

    print("Generating P-P Plot...")
    plt.figure(figsize=(8, 8))
    
    # Theoretical cumulative values
    theoretical_cdf = np.linspace(0, 1, num_test_events)

    # Plot 1-sigma, 2-sigma, and 3-sigma confidence intervals
    for ci, alpha, color in zip([0.68, 0.95, 0.997], [0.3, 0.2, 0.1], ['gray', 'lightgray', 'whitesmoke']):
        lower = binom.ppf((1 - ci) / 2, num_test_events, theoretical_cdf) / num_test_events
        upper = binom.ppf(1 - (1 - ci) / 2, num_test_events, theoretical_cdf) / num_test_events
        plt.fill_between(theoretical_cdf, lower, upper, color=color)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5) # Diagonal line for perfect calibration

    # Empirical CDF for each parameter
    colors = ['royalblue', 'forestgreen', 'crimson', 'darkorange', 'purple', 'teal']
    for j in range(num_params):
        sorted_percentiles = np.sort(percentiles[:, j])
        plt.plot(theoretical_cdf, sorted_percentiles, label=param_names[j], color=colors[j], linewidth=1.8)

    plt.xlabel('p')
    plt.ylabel('CDF(p)')
    plt.title(f'P-P Calibration Plot ($N={num_test_events}$ Test Events)')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='upper left', fontsize=9)
    
    pp_path = os.path.join(diagnostics_dir, "pp_plot.png")
    plt.savefig(pp_path, dpi=300, bbox_inches='tight')
    print(f"P-P Plot successfully saved to: '{pp_path}'")
    plt.show()

if __name__ == "__main__":
    run_pp_analysis()
