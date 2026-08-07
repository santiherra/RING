import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.flow import RingdownPosterior
from data.dataset import RingdownDataset
from utils.parser import load_config

def run_inference():
    '''
    Loads a trained RingdownPosterior model and performs inference on a single test waveform.
    Generates posterior samples and plots the results against the true parameters.
    '''

    # Load Config
    config = load_config("configs/config.yaml")
    
    # Resolve absolute path to guarantee finding the model inside the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(project_root, config['training']['save_dir'])

    # Device Configuration. Search for MPS (Apple Silicon), then CUDA, else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Loading model on {device} from {save_path}...")

    # Load the trained network using the architecture settings from config
    model = RingdownPosterior(
        context_dim=config['model']['context_dim'],
        hidden_features=config['model']['hidden_features'],
        num_transforms=config['model']['num_transforms']
    ).float().to(device)

    try:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: '{save_path}' not found. Did train.py finish successfully?")
        return
        
    model.eval()

    # Generate a single "blind" test waveform
    print("Generating a test waveform...")
    dataset = RingdownDataset(num_samples=1)
    
    signal_tensor, true_params = dataset[0]
    signal_tensor = signal_tensor.unsqueeze(0).to(device)

    # Generate Posterior Samples
    print("Running statistical inference...")
    with torch.no_grad():
        samples = model.sample(signal_tensor, num_samples=2000)

    samples = samples.squeeze(0).cpu().numpy()
    true_params = true_params.numpy()

    # Plot the Results in histograms for both remnant mass and spin
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(samples[:, 0], bins=30, density=True, color='royalblue', alpha=0.7, label='Network Posterior')
    axs[0].axvline(true_params[0], color='red', linestyle='dashed', linewidth=2, label='True Mass')
    axs[0].set_title("Predicted Remnant Mass ($M_f$)")
    axs[0].set_xlabel("Solar Masses")
    axs[0].set_ylabel("Probability Density")
    axs[0].legend()

    axs[1].hist(samples[:, 1], bins=30, density=True, color='seagreen', alpha=0.7, label='Network Posterior')
    axs[1].axvline(true_params[1], color='red', linestyle='dashed', linewidth=2, label='True Spin')
    axs[1].set_title("Predicted Dimensionless Spin ($a_f$)")
    axs[1].set_xlabel("Spin")
    axs[1].legend()

    plt.suptitle("Deep Learning Inference Results vs True Physics", fontsize=14)
    plt.tight_layout()
    plt.show()


''' QUICK TEST '''

if __name__ == "__main__":
    run_inference()
    