import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from models.embedding import WaveformEmbedding, ExtrinsicMLP

class RingdownPosterior(nn.Module):
    def __init__(self, sequence_length, model_config):
        """
        Combines the 1D CNN with a Normalizing Flow to predict parameter posteriors.
        
        Parameters:
        - sequence_length: The length of the input waveform.
        - model_config: A dictionary containing the model configuration parameters.
            - in_channels: Number of input channels (default 1).
            - param_dim: Dimensionality of the parameter space.
            - context_dim: Dimensionality of the context vector from the CNN.
            - hidden_features: Number of hidden features in the autoregressive transforms.
            - num_transforms: Number of autoregressive transforms to chain together in the flow.

        Returns:
        - A complete neural architecture that can be trained to predict posteriors from observed waveforms
        """

        # Initialize the RingdownPosterior Module
        super(RingdownPosterior, self).__init__()

        param_dim = model_config['param_dim']
        cnn_context_dim = model_config['context_dim']
        hidden_features = model_config['hidden_features']
        num_transforms = model_config['num_transforms']
        embedding_config = model_config['embedding_net']
        mlp_config = model_config['extrinsic_mlp']
        
        # Convolutional Neural Network object
        self.embedding_net = WaveformEmbedding(
            sequence_length=sequence_length, 
            embedding_config=embedding_config,
            output_dim=cnn_context_dim
        )

        # Multi-Layer Perceptron object
        self.extrinsic_mlp = ExtrinsicMLP(
            mlp_config=mlp_config
        )
        
        total_context_dim = cnn_context_dim + mlp_config['out_features']
        
        # Base distribution (Standard Normal)
        base_dist = StandardNormal(shape=[param_dim])
        
        # Autoregressive transforms
        transforms = []
        for _ in range(num_transforms):
            transforms.append(ReversePermutation(features=param_dim))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=param_dim, 
                    hidden_features=hidden_features, 
                    context_features=total_context_dim
                )
            )
        
        transform = CompositeTransform(transforms)
        
        # The final flow object
        self.flow = Flow(transform, base_dist)
        
    def forward(self, strain_data, target_parameters, extrinsic_params):
        """
        Used during TRAINING. 
        Calculates the Negative Log-Likelihood (NLL) of the true parameters given the data.

        Parameters:
        - strain_data: A batch of observed waveforms (shape: [batch_size, in_channels, sequence_length])
        - target_parameters: The true physical parameters corresponding to the waveforms
        - extrinsic_params: A batch of extrinsic physical parameters (shape: [batch_size, in_features])

        Returns:
        - The mean negative log-likelihood across the batch. Serves as loss function.
        """

        # Compress the 256 time steps into 64 features
        cnn_context = self.embedding_net(strain_data)
        mlp_context = self.extrinsic_mlp(extrinsic_params)

        context = torch.cat([cnn_context, mlp_context], dim=1)
        
        # Ask the flow for the log probability of the true parameters based on the context
        log_prob = self.flow.log_prob(inputs=target_parameters, context=context)
        
        return -log_prob.mean()

    def sample(self, strain_data, extrinsic_params, num_samples=1000):
        """
        Generates predicted parameter combinations for an observed waveform in inference. 

        Parameters:
        - strain_data: A single observed waveform (shape: [1, in_channels, sequence_length])
        - num_samples: The number of posterior samples to generate (default 1000)
        - extrinsic_params: Set of extrinsic physical parameters (shape: [1, in_features])

        Returns:
        - A tensor of shape [num_samples, param_dim] containing the predicted parameter combinations.
        """
        with torch.no_grad():
            cnn_context = self.embedding_net(strain_data)
            mlp_context = self.extrinsic_mlp(extrinsic_params)

            context = torch.cat([cnn_context, mlp_context], dim=1)

            samples = self.flow.sample(num_samples, context=context)
        return samples


''' QUICK TEST '''
if __name__ == "__main__":
    test_len = 256
    dummy_strain = torch.randn(32, 1, test_len)
    dummy_params = torch.randn(32, 6)
    dummy_extrinsics = torch.randn(32, 5)
    
    test_config = {
        'param_dim': 6, 'context_dim': 64, 'hidden_features': 128, 'num_transforms': 5,
        'extrinsic_mlp': {
            'in_features': 5, 'hidden_features': [16, 32], 'out_features': 32
        },
        'embedding_net': {
            'in_channels': 1, 'channels': [16, 32, 64],
            'kernel_size': 9, 'padding': 4, 'stride': 1, 'dilation': 1,
            'pool_size': 2, 'pool_stride': 2, 'pool_padding': 0, 'pool_dilation': 1
        }
    }
    
    model = RingdownPosterior(sequence_length=test_len, model_config=test_config)
    loss = model(dummy_strain, dummy_params, dummy_extrinsics)
    print(f"Calculated training loss (NLL): {loss.item():.4f}")
    