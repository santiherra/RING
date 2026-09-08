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
        
        # Load configuration dictionary
        if config is None:
            config = load_config(config_path)
        self.config = config
        
        # Physics parameters
        self.sample_rate = config['physics']['sample_rate']
        self.duration = config['physics']['duration']
        self.scale = float(config['physics']['scale'])
        
        # Fixed extrinsic parameters
        self.fixed_ra = config['event']['fixed_ra']
        self.fixed_dec = config['event']['fixed_dec']
        self.fixed_psi = config['event']['fixed_psi']
        self.fixed_gps = config['event']['fixed_gps']

        # Prior boundaries
        self.mass_min = config['priors']['mass_min']
        self.mass_max = config['priors']['mass_max']
        self.spin_min = config['priors']['spin_min']
        self.spin_max = config['priors']['spin_max']
        self.amp_min = config['priors']['amp_min']
        self.amp_max = config['priors']['amp_max']

        # Initialize PSD object
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
        
        # Sample target intrinsic parameters
        Mf = np.random.uniform(self.mass_min, self.mass_max)
        af = np.random.uniform(self.spin_min, self.spin_max)
        
        # Cartesian quadratures
        x_p_0 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        y_p_0 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        x_c_0 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        y_c_0 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        
        x_p_1 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        y_p_1 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        x_c_1 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        y_c_1 = np.random.uniform(self.amp_min, self.amp_max) * self.scale
        
        t, hp, hx = generate_ringdown(
            Mf=Mf, af=af, 
            x_p_0=x_p_0, y_p_0=y_p_0, x_c_0=x_c_0, y_c_0=y_c_0,
            x_p_1=x_p_1, y_p_1=y_p_1, x_c_1=x_c_1, y_c_1=y_c_1,
            sample_rate=self.sample_rate, duration=self.duration
        )
        
        h_det = project_to_detector(
            hp=hp, hx=hx, 
            ra=self.fixed_ra, dec=self.fixed_dec, psi=self.fixed_psi, gps_time=self.fixed_gps, 
            detector="H1"
        )

        compute_snr = self.config.get('diagnostics', {}).get('compute_snr', True)
        if compute_snr:
            h_fd = np.fft.rfft(h_det) * (1.0 / self.sample_rate)
            freqs = np.fft.rfftfreq(len(h_det), d=1.0 / self.sample_rate)
            psd_interp = self.psd.power_spectral_density_interpolated(freqs)
            valid_idx = psd_interp > 0
            snr_sq = 0.0
            if np.any(valid_idx):
                df = 1.0 / self.duration
                snr_sq = 4.0 * np.sum((np.abs(h_fd[valid_idx])**2) / psd_interp[valid_idx]) * df
            optimal_snr = np.sqrt(snr_sq)
        else:
            optimal_snr = -1.0
        
        noise = generate_noise(duration=self.duration, sample_rate=self.sample_rate, psd=self.psd)
        
        noisy_signal = h_det + noise
        scaled_signal = noisy_signal * (1.0 / self.scale)
        
        signal_tensor = torch.tensor(scaled_signal, dtype=torch.float32).unsqueeze(0)
        
        params_tensor = torch.tensor([
            Mf, af, 
            x_p_0 / self.scale, y_p_0 / self.scale, x_c_0 / self.scale, y_c_0 / self.scale,
            x_p_1 / self.scale, y_p_1 / self.scale, x_c_1 / self.scale, y_c_1 / self.scale
        ], dtype=torch.float32)

        snr_tensor = torch.tensor([optimal_snr], dtype=torch.float32)
        
        return signal_tensor, params_tensor, snr_tensor


''' QUICK TEST '''

if __name__ == "__main__":
    dataset = RingdownDataset(num_samples=10)
    signal, parameters, snr = dataset[0]
    print(f"Signal tensor shape: {signal.shape}")
    print(f"Parameters tensor shape: {parameters.shape}")
    print(f"True Remnant Mass: {parameters[0]:.2f} Solar Masses")
    print("Dataset successfully configured via configs/config.yaml.")
