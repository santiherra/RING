import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, ks_1samp, uniform
import bilby
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)

if project_root not in sys.path:
    sys.path.insert(0, project_root)    

from models.flow import RingdownPosterior
from data.dataset import RingdownDataset
from physics.transforms import get_gpu_noise_and_whiten
from physics.detector import get_detector_responses
from utils.parser import load_config
import argparse
import time

def run_pp_analysis():
    '''
    Run P-P plot analysis for the RingdownPosterior model over a set of test events.
    '''
    # Load configuration
    parser = argparse.ArgumentParser(description="Run P-P Plot Calibration.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                        help="Path to the configuration file relative to project root.")
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    pp_config = config.get('diagnostics', {}).get('pp_plot', {})
    num_test_events = pp_config.get('num_test_events', 1000)
    num_samples_per_event = pp_config.get('num_samples_per_event', 10000)
    
    save_path = os.path.join(project_root, config['training']['save_dir'])
    diagnostics_dir = os.path.join(project_root, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Device configuration. Search for MPS, then CUDA, else CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running P-P plot analysis on {device}\n")

    # Load model
    seq_len = int(config['physics']['duration'] * config['physics']['sample_rate'])
    scale = float(config['physics']['scale'])

    detector_names = config['event'].get('detectors', ['H1', 'L1'])
    config['model']['embedding_net']['in_channels'] = len(detector_names)

    model = RingdownPosterior(
        sequence_length=seq_len,
        model_config=config['model']
    ).float().to(device)
    
    try:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: '{save_path}' not found. Please run train.py first.")
        return
        
    model.eval()

    # Test dataset
    print(f"Loading test dataset: {num_test_events} waveforms...")
    t0_data = time.time()
    test_dataset = RingdownDataset(num_samples=num_test_events, dataset_type='test')
    print(f"Test waveforms loaded in {time.time() - t0_data:.2f} s.")

    responses = get_detector_responses(
        detector_names=detector_names,
        ra=config['event']['fixed_ra'],
        dec=config['event']['fixed_dec'],
        psi=config['event']['fixed_psi'],
        gps_time=config['event']['fixed_gps'],
        sample_rate=config['physics']['sample_rate'],
        duration=config['physics']['duration']
    )

    F_plus = torch.tensor(responses['F_plus'], dtype=torch.float32, device=device)
    F_cross = torch.tensor(responses['F_cross'], dtype=torch.float32, device=device)
    time_delays = torch.tensor(responses['time_delay'], dtype=torch.float32, device=device)
    asd_tensors = torch.tensor(responses['asd'], dtype=torch.float32, device=device)
    
    # Target parameter names
    param_names = [r"$M_f$", r"$a_f$", 
                   r"$x_{+,0}$", r"$y_{+,0}$", r"$x_{\times,0}$", r"$y_{\times,0}$",
                   r"$x_{+,1}$", r"$y_{+,1}$", r"$x_{\times,1}$", r"$y_{\times,1}$"]
    num_params = len(param_names)
    
    # Matrix of percentile ranks: shape [num_test_events, num_params]
    checkpoint_path = os.path.join(diagnostics_dir, "pp_checkpoint.npz")
    percentiles = np.zeros((num_test_events, num_params))
    start_event = 0

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint. Resuming P-P plot...")
        chkpt = np.load(checkpoint_path)
        percentiles = chkpt['percentiles']
        start_event = chkpt['last_event'] + 1
        print(f"Resuming from event [{start_event + 1}/{num_test_events}]...")

    max_runtime = pp_config.get('max_runtime_minutes', float('inf')) * 60
    global_start_time = time.time()
    batch_start_time = time.time()

    # Loop over all test events
    print("Evaluating posteriors...\n")
    try:
        for i in range(start_event, num_test_events):
            hp_tensor, hx_tensor, true_params = test_dataset[i]
            
            hp_tensor = hp_tensor.unsqueeze(0).to(device)
            hx_tensor = hx_tensor.unsqueeze(0).to(device)
            true_params = true_params.numpy()

            signal_tensor, _ = get_gpu_noise_and_whiten(
                hp_tensor, hx_tensor, F_plus, F_cross, time_delays, asd_tensors, scale, 
                config['physics']['sample_rate'], config['physics']['duration'], device
            )
    
            with torch.no_grad():
                samples = model.sample(signal_tensor, num_samples=num_samples_per_event)
                samples = samples.squeeze(0).cpu().numpy()
    
            for j in range(num_params):
                rank = np.sum(samples[:, j] <= true_params[j]) / num_samples_per_event
                percentiles[i, j] = rank
            
            np.savez(checkpoint_path, percentiles=percentiles, last_event=i)
    
            if (i + 1) % 20 == 0 or (i + 1) == num_test_events:
                elapsed_batch = time.time() - batch_start_time
                elapsed_total = time.time() - global_start_time
                events_in_batch = 20 if (i + 1) % 20 == 0 else (i + 1) % 20
                print(f"Events [{i + 1}/{num_test_events}].  Last {events_in_batch} events: {elapsed_batch:.1f} s.  Total: {elapsed_total:.1f} s")
                batch_start_time = time.time() 
                
            if time.time() - global_start_time > max_runtime:
                print(f"\nWARNING: Time limit of {max_runtime/60:.1f} minutes reached. Exiting.")
                sys.exit(0)

    except KeyboardInterrupt:
        print("\nWARNING: P-P plot evaluation interrupted. Progress saved to checkpoint.")
        sys.exit(0)

    print("\nGenerating P-P plot...")
    plt.figure(figsize=(8, 8))
    
    # Theoretical cumulative values
    theoretical_cdf = np.linspace(0, 1, num_test_events)

    # Confidence intervals for 1-sigma, 2-sigma, and 3-sigma
    for ci, alpha, color in zip([0.68, 0.95, 0.997], [0.3, 0.2, 0.1], ['gray', 'lightgray', 'whitesmoke']):
        lower = binom.ppf((1 - ci) / 2, num_test_events, theoretical_cdf) / num_test_events
        upper = binom.ppf(1 - (1 - ci) / 2, num_test_events, theoretical_cdf) / num_test_events
        plt.fill_between(theoretical_cdf, lower, upper, color=color, alpha=alpha)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5) # Diagonal line

    # Parameter empirical CDFs
    colors = ['royalblue', 'forestgreen', 'crimson', 'darkorange', 
              'purple', 'teal', 'magenta', 'gold', 'brown', 'navy']
    for j in range(num_params):
        sorted_percentiles = np.sort(percentiles[:, j])
        
        # Kolmogorov-Smirnov against a Uniform(0,1) distribution
        ks_stat = ks_1samp(percentiles[:, j], uniform(0, 1).cdf)
        p_val = ks_stat.pvalue
        
        label_with_pval = f"{param_names[j]} (p={p_val:.3f})"
        
        plt.plot(theoretical_cdf, sorted_percentiles, label=label_with_pval, color=colors[j], linewidth=1.8)

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
