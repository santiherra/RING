import os
import sys
import time
import h5py
import numpy as np
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from physics.waveform import generate_ringdown
from utils.parser import load_config
import argparse

def generate_hdf5_dataset(output_file, num_samples, config):
    '''
    Generates a clean dataset of ringdown polarizations and saves to HDF5.

    Params: 
    - output_file: Path to store HDF5 file.
    - num_samples: Number of samples to generate.
    - config: Configuration dictionary containing physics parameters.
    '''
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    sample_rate = config['physics']['sample_rate']
    duration = config['physics']['duration']
    scale = float(config['physics']['scale'])
    
    mass_min = config['priors']['mass_min']
    mass_max = config['priors']['mass_max']
    spin_min = config['priors']['spin_min']
    spin_max = config['priors']['spin_max']
    amp_min = config['priors']['amp_min']
    amp_max = config['priors']['amp_max']

    seq_len = int(duration * sample_rate)

    print(f"Creating HDF5 file at: {output_file}")
    t0 = time.time()
    
    with h5py.File(output_file, 'w') as f:
        hp_dset = f.create_dataset('hp', shape=(num_samples, seq_len), dtype='float32')
        hx_dset = f.create_dataset('hx', shape=(num_samples, seq_len), dtype='float32')
        params_dset = f.create_dataset('params', shape=(num_samples, 10), dtype='float32')

        for i in range(num_samples):
            Mf = np.random.uniform(mass_min, mass_max)
            af = np.random.uniform(spin_min, spin_max)

            x_p_0 = np.random.uniform(amp_min, amp_max) * scale
            y_p_0 = np.random.uniform(amp_min, amp_max) * scale
            x_c_0 = np.random.uniform(amp_min, amp_max) * scale
            y_c_0 = np.random.uniform(amp_min, amp_max) * scale

            x_p_1 = np.random.uniform(amp_min, amp_max) * scale
            y_p_1 = np.random.uniform(amp_min, amp_max) * scale
            x_c_1 = np.random.uniform(amp_min, amp_max) * scale
            y_c_1 = np.random.uniform(amp_min, amp_max) * scale

            _, hp, hx = generate_ringdown(
                Mf=Mf, af=af,
                x_p_0=x_p_0, y_p_0=y_p_0, x_c_0=x_c_0, y_c_0=y_c_0,
                x_p_1=x_p_1, y_p_1=y_p_1, x_c_1=x_c_1, y_c_1=y_c_1,
                sample_rate=sample_rate, duration=duration
            )

            hp_dset[i] = hp
            hx_dset[i] = hx
            params_dset[i] = [
                Mf, af,
                x_p_0 / scale, y_p_0 / scale, x_c_0 / scale, y_c_0 / scale,
                x_p_1 / scale, y_p_1 / scale, x_c_1 / scale, y_c_1 / scale
            ]

            if (i + 1) % 5000 == 0 or (i + 1) == num_samples:
                print(f"Generated [{i + 1}/{num_samples}] waveforms.")

    print(f"Dataset generated in {time.time() - t0:.2f} s.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HDF5 datasets.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(os.path.join(project_root, args.config))

    train_file = os.path.join(project_root, cfg['data']['train_file'])
    val_file = os.path.join(project_root, cfg['data']['val_file'])
    test_file = os.path.join(project_root, cfg['data']['test_file'])

    num_train = cfg['training']['num_train_samples']
    num_val = cfg['training']['num_val_samples']
    num_test = cfg['diagnostics']['pp_plot']['num_test_events']

    generate_hdf5_dataset(train_file, num_train, cfg)
    generate_hdf5_dataset(val_file, num_val, cfg)
    generate_hdf5_dataset(test_file, num_test, cfg)
