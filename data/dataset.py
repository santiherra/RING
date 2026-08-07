import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset
import numpy as np
import bilby
import logging

# Mute Bilby's terminal logging
logging.getLogger('bilby').setLevel(logging.WARNING)

from physics.waveform import generate_ringdown
from physics.detector import project_to_detector
from physics.noise import generate_noise
from utils.parser import load_config

class RingdownDataset(Dataset):
    def __init__(self, num_samples, config=None, config_path="configs/config.yaml"):
        '''
        Generates synthetic ringdown waveforms with noise, based on specified physics 
        parameters and configuration.

        Parameters:
        - num_samples: Number of samples to generate in the dataset
        - config: Optional configuration dictionary. If None, will load from config_path.
        - config_path: Path to the YAML configuration file (default: "configs/config.yaml")

        Returns:
        - Initializes the dataset object with specified parameters and configuration.
        '''

        self.num_samples = num_samples
        
        # Load configuration dictionary if not explicitly provided
        if config is None:
            config = load_config(config_path)
        self.config = config
        
        # Physics parameters from config
        self.sample_rate = config['physics']['sample_rate']
        self.duration = config['physics']['duration']
        self.scale = float(config['physics']['scale'])
        
        # Fixed Extrinsic Parameters from config (Anchored to GW250114)
        self.fixed_iota = config['event']['fixed_iota']
        self.fixed_phi = config['event']['fixed_phi']
        self.fixed_ra = config['event']['fixed_ra']
        self.fixed_dec = config['event']['fixed_dec']
        self.fixed_psi = config['event']['fixed_psi']
        self.fixed_gps = config['event']['fixed_gps']

        # Initialize PSD object once
        self.psd = bilby.gw.detector.PowerSpectralDensity.from_aligo()

    def __len__(self):
        '''
        Returns the number of samples in the dataset.
        '''

        return self.num_samples

    def __getitem__(self, idx):
        '''
        Generates a sample of the dataset, consisting of a noisy ringdown waveform
        and its corresponding intrinsic parameters.

        Parameters:
        - idx: Index of the sample to generate (not used for random generation)

        Returns:
        - signal_tensor: PyTorch tensor of the noisy ringdown signal (shape: [1, num_samples])
        - params_tensor: PyTorch tensor of the intrinsic parameters
                         (shape: [6], containing [Mf, af, C_re_0, C_im_0, C_re_1, C_im_1])
        '''
        
        # Randomly sample target intrinsic parameters
        Mf = np.random.uniform(50.0, 100.0)
        af = np.random.uniform(0.5, 0.99)
        
        # Cartesian amplitudes generated at physical scale from config
        C_re_0 = np.random.uniform(-5.0, 5.0) * self.scale
        C_im_0 = np.random.uniform(-5.0, 5.0) * self.scale
        C_re_1 = np.random.uniform(-5.0, 5.0) * self.scale
        C_im_1 = np.random.uniform(-5.0, 5.0) * self.scale
        
        # Generate physics waveform
        t, hp, hx = generate_ringdown(
            Mf=Mf, af=af, 
            C_re_0=C_re_0, C_im_0=C_im_0, C_re_1=C_re_1, C_im_1=C_im_1, 
            iota=self.fixed_iota, phi=self.fixed_phi, 
            sample_rate=self.sample_rate, duration=self.duration
        )
        
        # Project to detector
        h_det = project_to_detector(
            hp=hp, hx=hx, 
            ra=self.fixed_ra, dec=self.fixed_dec, psi=self.fixed_psi, gps_time=self.fixed_gps, 
            detector="H1"
        )
        
        # Generate colored noise
        noise = generate_noise(duration=self.duration, sample_rate=self.sample_rate, psd=self.psd)
        
        # Combine and scale for PyTorch
        noisy_signal = h_det + noise
        scaled_signal = noisy_signal * (1.0 / self.scale)
        
        # Format Tensors
        signal_tensor = torch.tensor(scaled_signal, dtype=torch.float32).unsqueeze(0)
        
        params_tensor = torch.tensor(
            [Mf, af, C_re_0 / self.scale, C_im_0 / self.scale, C_re_1 / self.scale, C_im_1 / self.scale], 
            dtype=torch.float32
        )
        
        return signal_tensor, params_tensor


''' QUICK TEST '''

if __name__ == "__main__":
    dataset = RingdownDataset(num_samples=10)
    signal, parameters = dataset[0]
    print(f"Signal tensor shape: {signal.shape}")
    print(f"Parameters tensor shape: {parameters.shape}")
    print(f"True Remnant Mass: {parameters[0]:.2f} Solar Masses")
    print("Dataset successfully configured via configs/config.yaml.")
