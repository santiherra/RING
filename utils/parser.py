import os
import yaml

def load_config(config_path="configs/config.yaml"):
    """
    Reads the YAML configuration file and returns a Python dictionary.
    Dynamically resolves relative paths against the project root directory.

    Parameters:
    - config_path: Path to the YAML configuration file (default "configs/config.yaml").

    Returns:
    - A dictionary containing the configuration parameters.
    """

    # If the path is relative, resolve it against the project root
    if not os.path.isabs(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_path)

    # Check if the configuration file exists. If not, raise an error
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: '{config_path}'. "
            "Please ensure 'configs/config.yaml' exists in your main project folder."
        )

    # Load the YAML configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    return config
