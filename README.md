# RING

**RING** is a deep learning pipeline coded in Python designed to perform fast, highly accurate parameter estimation on the ring-down phase of Binary Black Holes (BBH). By employing a Conditional Masked Autoregressive Flow (MAF) alongside a 1D Convolutional Neural Network feature extractor, this repository aims to rapidly determine the final mass ($M_f$) and dimensionless spin ($a_f$) of remnant black holes to facilitate tests of General Relativity.

## Installation

To run RING, you will need to clone the repository and it is convenient to set up a Python virtual environment.

### For Mac/Linux:
1. Open your terminal and navigate to the project folder:
   ```sh
   cd path/to/RING
   ```

2. Create and activate the virtual environment:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required dependencies
    ```sh
    pip install -r requirements.txt
    ````

### For Windows
1. Open Command Prompt or PowerShell and navigate to the project folder:
    ``sh
    cd path\to\RING
    ```

2. Create and activate the virtual environment:
    ```sh
    python -m venv .venv
    .venv\Scripts\activate
    ````

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ````

(Note: Whenever you want to run the training or inference scripts, ensure you have activated the virtual environment first. You will know it is active when `(.venv)` appears at the beginning of your terminal prompt).

## Project Structure & Contents

The repository is modularly designed, separating the physics generation, deep learning architecture, and execution logic.

1. Master Execution Scripts
    * `train.py`: The primary executable for training the model. It reads the configuration file, generates the physics dataset, initializes the neural network, executes the training loop, and saves the best model weights. Run this via `python train.py`.
    * `inference.py`: The primary executable for evaluating the model. It loads a pre-trained model, generates a blind test waveform, runs statistical inference, and outputs a plot comparing the network's predicted posterior distributions against the true hidden parameters. Run this via `python inference.py`.

2. Core Directories
    * `configs/`
        * `config.yaml`: The master control file. Use this to alter training hyperparameters (learning rate, epochs, batch size), neural network depth, physics duration, and model save paths without ever touching the Python code.
    * `physics/`
        * `waveform.py: Generates the pure, mathematical ringdown waveforms using quasinormal modes (QNM).
        * detector.py: Projects the pure waveform onto the Advanced LIGO (H1) detector antenna pattern based on sky location and GPS time.
        * noise.py: Generates colored Gaussian noise using the aLIGO Power Spectral Density (PSD).
    * data/
        * dataset.py: The PyTorch Dataset factory. It randomly samples remnant mass and spin, calls the physics module to generate a noisy strain, and scales the final tensor for the neural network.
        * dataloader.py: Wraps the dataset into PyTorch DataLoader objects to feed data in batches to the GPU/CPU during training.
    * models/
        * embedding.py: Contains a 1D Convolutional Neural Network (CNN). It acts as a feature extractor, compressing the 256 time-steps of the waveform into a dense vector of 64 summary features.
        * flow.py: The brain of the network. It utilizes nflows to build a Normalizing Flow that learns the complex posterior probability distribution of the parameters conditioned on the CNN's summary features.
    * utils/
        * parser.py: A utility script that safely loads and parses config.yaml using dynamic pathing, ensuring the configuration can be read regardless of where the terminal is executed.
    * saved_models/
    This folder is automatically generated during training. It serves as the storage location for your saved .pth model weights.

## References & Dependencies

This project relies on several open-source libraries. For further reading on the underlying mechanics, refer to their official documentation:
    * PyTorch: The core deep learning framework used for building the CNN and handling tensor operations.
    * Bilby: Used for accessing the advanced LIGO Power Spectral Density (PSD) and generating realistic detector noise.
    * nflows: A suite of Normalizing Flows built in PyTorch, heavily utilized here to map normal distributions into the complex BBH parameter posteriors.
    * NumPy & Matplotlib: Standard libraries for mathematical operations and results visualization.

