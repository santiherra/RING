import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.parser import load_config
from physics.waveform import generate_ringdown

class RingdownDataset(Dataset):
    def __init__(self, num_samples, file_path=None, config=None, config_path="configs/config.yaml", dataset_type="train"):
        '''
        Generates synthetic ringdown waveforms with noise, based on specified physics 
        parameters and configuration. Stores in an HDF5 file for training, validation, or performance diagnostics, 
        or generates a single injection for inference of an event.

        Parameters:
        - num_samples: Number of samples to generate in the dataset
        - file_path: Optional path to save the dataset HDF5 file. If None, will use default paths based on dataset_type.
        - config: Optional configuration dictionary. If None, will load from config_path.
        - config_path: Path to the YAML configuration file (default: "configs/config.yaml")

        Returns:
        - Saves the dataset HDF5 file or single injection object with specified parameters and configuration.
        '''

        self.num_samples = num_samples
        self.dataset_type = dataset_type
        
        # Load configuration dictionary
        if config is None:
            config = load_config(config_path)
        self.config = config
        
        if self.dataset_type == "inference":
            self._generate_inference_data()
        else:
            if file_path is None:
                if dataset_type == "train":
                    self.file_path = os.path.join(project_root, config['data']['train_file'])
                elif dataset_type == "val":
                    self.file_path = os.path.join(project_root, config['data']['val_file'])
                else:
                    self.file_path = os.path.join(project_root, config['data']['test_file'])
            else:
                self.file_path = file_path

            self.file = None
            with h5py.File(self.file_path, 'r') as f:
                self.length = len(f['params'])

    def _generate_inference_data(self):
        '''
        Generates a single injection for event inference.
        '''
        scale = float(self.config['physics']['scale'])
        Mf = np.random.uniform(self.config['priors']['mass_min'], self.config['priors']['mass_max'])
        af = np.random.uniform(self.config['priors']['spin_min'], self.config['priors']['spin_max'])
        
        x_p_0 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale
        y_p_0 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale
        x_c_0 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale
        y_c_0 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale
        
        x_p_1 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale
        y_p_1 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale
        x_c_1 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale
        y_c_1 = np.random.uniform(self.config['priors']['amp_min'], self.config['priors']['amp_max']) * scale

        _, hp, hx = generate_ringdown(
            Mf=Mf, af=af, 
            x_p_0=x_p_0, y_p_0=y_p_0, x_c_0=x_c_0, y_c_0=y_c_0,
            x_p_1=x_p_1, y_p_1=y_p_1, x_c_1=x_c_1, y_c_1=y_c_1,
            sample_rate=self.config['physics']['sample_rate'], duration=self.config['physics']['duration']
        )

        params = np.array([
            Mf, af, 
            x_p_0 / scale, y_p_0 / scale, x_c_0 / scale, y_c_0 / scale,
            x_p_1 / scale, y_p_1 / scale, x_c_1 / scale, y_c_1 / scale
        ])

        self.inf_hp = torch.tensor(hp, dtype=torch.float32)
        self.inf_hx = torch.tensor(hx, dtype=torch.float32)
        self.inf_params = torch.tensor(params, dtype=torch.float32)

    def __len__(self):
        '''
        Returns:
        - Training/validation/testing: number of samples in the dataset
        - Inference: number of samples to generate
        '''
        if self.dataset_type == "inference":
            return self.num_samples
        return self.length

    def __getitem__(self, idx):
        '''
        Generates a sample of the dataset, consisting of a noisy ringdown waveform
        and its corresponding intrinsic parameters.

        Parameters:
        - idx: Index of the sample to generate

        Returns:
        - Training/validation/testing:
          - signal_tensor: PyTorch tensor of the noisy ringdown signal (shape: [1, num_samples])
          - params_tensor: PyTorch tensor of the intrinsic parameters
                           (shape: [10], containing [Mf, af, C_re_0, C_im_0, C_re_1, C_im_1])
        - Inference:
            - inf_hp: plus polarization waveform tensor (shape: [num_samples])
            - inf_hx: cross polarization waveform tensor (shape: [num_samples])
            - inf_params: intrinsic parameters tensor (shape: [10] by default)
        '''
        if self.dataset_type == "inference":
            return self.inf_hp, self.inf_hx, self.inf_params
        
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        hp = self.file['hp'][idx]
        hx = self.file['hx'][idx]
        params = self.file['params'][idx]

        hp_tensor = torch.tensor(hp, dtype=torch.float32)
        hx_tensor = torch.tensor(hx, dtype=torch.float32)
        params_tensor = torch.tensor(params, dtype=torch.float32)

        return hp_tensor, hx_tensor, params_tensor


''' QUICK TEST '''

if __name__ == "__main__":
    dataset = RingdownDataset(num_samples=10)
    hp, hx, params = dataset[0]
    print(f"hp tensor shape: {hp.shape}")
    print(f"hx tensor shape: {hx.shape}")
    print(f"Parameters tensor shape: {params.shape}")
    print(f"True Remnant Mass: {params[0]:.2f} Solar Masses")
    print("Dataset successfully configured via configs/config.yaml.")
