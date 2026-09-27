import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from data.dataset import RingdownDataset
from utils.parser import load_config

def get_dataloaders(num_train=None, num_val=None, batch_size=None, config_path="configs/config.yaml"):
    """
    Creates PyTorch DataLoaders configured via config.yaml.

    Parameters:
    - num_train: Number of training samples (overrides config if provided)
    - num_val: Number of validation samples (overrides config if provided)
    - batch_size: Batch size for DataLoaders (overrides config if provided)
    - config_path: Path to the YAML configuration file (default: "configs/config.yaml")

    Returns:
    - train_loader: DataLoader for the training dataset
    - val_loader: DataLoader for the validation dataset
    """

    # Load configuration dictionary
    config = load_config(config_path)
    
    # Config values
    if num_train is None:
        num_train = config['training']['num_train_samples']
    if num_val is None:
        num_val = config['training']['num_val_samples']
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    # Initialize dataset with config
    train_dataset = RingdownDataset(num_samples=num_train, config=config, dataset_type='train')
    val_dataset = RingdownDataset(num_samples=num_val, config=config, dataset_type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


''' QUICK TEST '''

if __name__ == "__main__":
    train_dl, val_dl = get_dataloaders()
    for hp_b, hx_b, params_b in train_dl:
        print(f"Batch hp Tensor Shape: {hp_b.shape}") 
        print(f"Batch hx Tensor Shape: {hx_b.shape}") 
        print(f"Batch Parameters Tensor Shape: {params_b.shape}")
        break
    print("DataLoaders successfully configured via configs/config.yaml.")
