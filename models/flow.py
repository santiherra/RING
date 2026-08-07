import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

# nflows imports for the Normalizing Flow architecture
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from models.embedding import WaveformEmbedding

class RingdownPosterior(nn.Module):
    def __init__(self, param_dim=6, context_dim=64, hidden_features=128, num_transforms=5):
        """
        Combines the 1D CNN with a Normalizing Flow to predict parameter posteriors.
        
        Parameters:
        - param_dim: The number of physical parameters to predict (default 6).
        - context_dim: The number of features extracted by the CNN (default 64).
        - hidden_features: Width of the internal neural networks inside the flow (default 128).
        - num_transforms: Number of flow layers - depth of the statistical engine (default 5).

        Returns:
        - A complete neural architecture that can be trained to predict posteriors from observed waveforms
        """

        # Initialize the RingdownPosterior Module
        super(RingdownPosterior, self).__init__()
        
        # The Feature Extractor (CNN)
        self.embedding_net = WaveformEmbedding(in_channels=1, output_dim=context_dim)
        
        # Build the Normalizing Flow. The base distribution is a simple 6D Gaussian
        base_dist = StandardNormal(shape=[param_dim])
        
        # Chain together multiple Autoregressive transforms to warp the Gaussian
        transforms = []
        for _ in range(num_transforms):
            transforms.append(ReversePermutation(features=param_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=param_dim, 
                    hidden_features=hidden_features, 
                    context_features=context_dim
                )
            )
        
        transform = CompositeTransform(transforms)
        
        # The final flow object
        self.flow = Flow(transform, base_dist)
        
    def forward(self, strain_data, target_parameters):
        """
        Used during TRAINING. 
        Calculates the Negative Log-Likelihood (NLL) of the true parameters given the data.

        Parameters:
        - strain_data: A batch of observed waveforms (shape: [batch_size, 1, 256])
        - target_parameters: The true physical parameters corresponding to the waveforms

        Returns:
        - The mean negative log-likelihood across the batch, which serves as the loss for optimization
        """

        # Compress the 256 time steps into 64 features
        context = self.embedding_net(strain_data)
        
        # Ask the flow for the log probability of the true parameters based on the context
        log_prob = self.flow.log_prob(inputs=target_parameters, context=context)
        
        return -log_prob.mean()

    def sample(self, strain_data, num_samples=1000):
        """
        Used during INFERENCE.
        Generates predicted parameter combinations for an observed waveform.

        Parameters:
        - strain_data: A single observed waveform (shape: [1, 1, 256])
        - num_samples: The number of posterior samples to generate (default 1000)

        Returns:
        - A tensor of shape [num_samples, param_dim] containing the predicted parameter combinations
        """
        with torch.no_grad():
            context = self.embedding_net(strain_data)
            samples = self.flow.sample(num_samples, context=context)
        return samples

''' QUICK TEST '''
if __name__ == "__main__":
    # Simulate a batch of 32 observed waveforms and their 32 true parameter sets
    dummy_strain = torch.randn(32, 1, 256)
    dummy_params = torch.randn(32, 6)
    
    # Initialize the complete brain
    model = RingdownPosterior()
    
    # Test the forward pass (Training Mode)
    loss = model(dummy_strain, dummy_params)
    print(f"Calculated Training Loss (NLL): {loss.item():.4f}")
    
    # Test the sample pass (Inference Mode on a single waveform)
    single_waveform = dummy_strain[0:1] # Take just the first waveform
    predicted_samples = model.sample(single_waveform, num_samples=500)
    print(f"Generated {predicted_samples.shape[1]} posterior samples for inference.")
    print("Full neural architecture is complete and ready for training!")
    