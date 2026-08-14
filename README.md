# RING

**RING** is a deep learning pipeline coded in Python designed to perform fast, highly accurate parameter estimation on the ring-down phase of Binary Black Holes (BBH). By employing a conditional Masked Autoregressive Flow (MAF) alongside a 1D Convolutional Neural Network feature extractor, this repository aims to rapidly determine the final mass ($M_f$) and dimensionless spin ($a_f$) of remnant black holes, as well as to resolve the modes contributing to the wave signal, to facilitate tests of General Relativity.

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
    ```

### For Windows
1. Open Command Prompt or PowerShell and navigate to the project folder:
    ```sh
    cd path\to\RING
    ```

2. Create and activate the virtual environment:
    ```sh
    python -m venv .venv
    .venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

(Note: Whenever you want to run the training or inference scripts, ensure you have activated the virtual environment first. You will know it is active when `(.venv)` appears at the beginning of your terminal prompt).

## Usage & Dynamic Configurations

All execution scripts in RING are designed to accept dynamic configuration files via the `--config` flag. This allows you to run multiple experiments (e.g., changing SNR ranges, epochs, or neural network depths) without overwriting your master settings. 

The configuration files are stored in the `config/` folder. The scripts default to `configs/config.yaml`. In order to produce and test different configurations, you may duplicate the file with modifications to the desired features. 

* Train a model:
    ```sh
    python train.py --config configs/config.yaml
    ```
* Run single-event inference:
    ```sh
    python evaluation/inference.py --config configs/config.yaml
    ```
* Run statistical calibration:
    ```sh
    python evaluation/pp_plot.py --config configs/config.yaml
    ```

(Note: if you would like to use a different configuration file to the default `config.yaml`, replace it by the name of the targeted one in the command prompts after `--config`).

## Project Structure & Contents

The repository is modularly designed, separating the physics generation, deep learning architecture, and execution logic.

1. Master Execution Scripts
    * `train.py`: The primary executable for training the model. It reads the configuration file, generates the physics dataset, initializes the neural network, executes the training loop, and saves the best model weights. Run this by typing the following command in the terminal,
    ```sh
    python train.py --config configs/config.yaml
    ```

2. Core Directories
    * `configs/`
        * `config.yaml`: The master control file. Use this to alter training hyperparameters (learning rate, epochs, batch size), neural network depth, physics duration, and model save paths without ever touching the Python code.
    * `evaluation/`
        * `inference.py`: The primary executable for evaluating the model. It loads a pre-trained model, generates a blind test waveform, runs statistical inference, and outputs a plot comparing the network's predicted posterior distributions against the true hidden parameters. Run this by typing the following command in the terminal,
        ```sh
        python evaluation/inference.py --config configs/config.yaml
        ```
        * `pp_plot.py`: Generates test waveforms to rigorously evaluate the statistical calibration of the network's uncertainties through Percentile-Percentile (P-P) plotting. Run this by typing the following command in the terminal,
        ```sh
        python evaluation/pp_plot.py --config configs/config.yaml
        ```
    * `diagnostics/`
    Auto-generated folder where the pipeline saves visual benchmarking tools, including Negative Log-Likelihood (NLL) learning curves, optimal matched filter SNR distributions, and P-P plots.
    * `physics/`
        * `waveform.py`: Generates the pure, mathematical ringdown waveforms using quasinormal modes (QNM).
        * `detector.py`: Projects the pure waveform onto the Advanced LIGO (H1) detector antenna pattern based on sky location and GPS time.
        * `noise.py`: Generates colored Gaussian noise using the aLIGO Power Spectral Density (PSD).
    * `data/`
        * `dataset.py`: The PyTorch Dataset factory. It randomly samples remnant mass and spin, calls the physics module to generate a noisy strain, and scales the final tensor for the neural network.
        * `dataloader.py`: Wraps the dataset into PyTorch DataLoader objects to feed data in batches to the GPU/CPU during training.
    * `models/`
        * `embedding.py`: Contains a 1D Convolutional Neural Network (CNN). It acts as a feature extractor, compressing the 256 time-steps of the waveform into a dense vector of 64 summary features.
        * `flow.py`: The brain of the network. It utilizes nflows to build a Normalizing Flow that learns the complex posterior probability distribution of the parameters conditioned on the CNN's summary features.
    * `utils/`
        * `parser.py`: A utility script that safely loads and parses config.yaml using dynamic pathing, ensuring the configuration can be read regardless of where the terminal is executed.
    * `saved_models/`
    This folder is automatically generated during training. It serves as the storage location for your saved .pth model weights.

## References & Dependencies

This project relies on several open-source libraries. For further reading on the underlying mechanics, refer to their official documentation:
* [Pytorch](https://pytorch.org): The core software used for building the CNN and handling tensor operations.
* [bilby](https://pypi.org/project/bilby/): Used for accessing the advanced LIGO Power Spectral Density (PSD) and generating realistic detector noise.
* [nflows](https://github.com/bayesiains/nflows): normalizing flows in PyTorch.
* [qnm](https://qnm.readthedocs.io/en/latest/README.html#): package for computing the complex frequencies andamplitude parameters for the QNM of a remnant black hole, as well as their decomposition in spheroidal harmonics.
* [NumPy](https://numpy.org/doc/2.1/index.html), [SciPy](https://scipy.org/es/) & [Matplotlib](https://matplotlib.org): Standard libraries for mathematical operations and results visualization.
* [corner](https://corner.readthedocs.io/en/latest/):Used to visualize multi-dimensional parameter distributions usingdense scatterplot matrices and contour intervals.

### References
* > Gregory Ashton et al., _BILBY: A user-friendly Bayesian inference library for gravitational-wave astronomy_, The Astrophysical Journal Supplement Series (2019) 241, 27. [[arXiv]](https://arxiv.org/abs/1811.02042)
* > Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios, _Neural Spline Flows_, NeurIPS 2019. [[arXiv]](https://arxiv.org/abs/1906.04032)
* > Leo C. Stein, _qnm: A Python package for calculating Kerr quasinormal modes, separation constants, and spherical-spheroidal mixing coefficients_, Journal of Open Source Software, 4(42), 1683 (2019). [[arXiv]](https://arxiv.org/abs/2002.03712)
* > Daniel Foreman-Mackey, _corner.py: Scatterplot matrices in Python_, Journal of Open Source Software, 1(2), 24 (2016). [[JOSS]](https://joss.theoj.org/papers/10.21105/joss.00024)
