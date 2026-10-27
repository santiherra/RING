import math

import torch
import torch.nn as nn

class WaveformEmbedding(nn.Module):
    def __init__(self, sequence_length, embedding_config, output_dim=64):
        '''
        Initialises a 1D Convolutional Neural Network that compresses a time-domain ringdown
        into a dense vector of summary features.
        
        Parameters:
        - sequence_length: The length of the input waveform.
        - embedding_config: A dictionary containing the CNN configuration parameters:
            - in_channels: Number of input channels (default 1).
            - channels: List of output channels for each convolutional layer.
            - kernel_size: Size of the convolutional kernel (default 9).
            - padding: Amount of zero-padding added to both sides of the input (default 4).
            - stride: Stride of the convolution (default 1).
            - dilation: Spacing between kernel elements (default 1).
            - pool_size: Size of kernel of the max pooling window (default 2).
            - pool_stride: Stride of the max pooling window (default 2).
            - pool_padding: Amount of zero-padding added to both sides of the input for pooling (default 0).
            - pool_dilation: Spacing between kernel elements for pooling (default 1).
        - output_dim: The number of summary features to output (default 64).
        '''
        super(WaveformEmbedding, self).__init__()

        self.in_channels = embedding_config.get('in_channels', 1)
        self.channels = embedding_config['channels']

        # Conv1d params
        self.k = embedding_config['kernel_size']
        self.p = embedding_config['padding']
        self.s = embedding_config['stride']
        self.d = embedding_config['dilation']
        
        # MaxPool1d params
        self.pk = embedding_config['pool_size']
        self.ps = embedding_config['pool_stride']
        self.pp = embedding_config['pool_padding']
        self.pd = embedding_config['pool_dilation']

        # Regularisation params
        self.use_dropout = embedding_config.get('use_dropout', False)
        self.dropout_rate = embedding_config.get('dropout_rate', 0.0)

        # Architectural consistency checker
        final_seq_length = self._check_algebra(sequence_length)
        flattened_size = final_seq_length * self.channels[-1]

        layers = []
        current_in_channels = self.in_channels
        
        # Build as many sequences as defined in config.yaml
        for out_channels in self.channels:
            layers.append(nn.Conv1d(
                in_channels=current_in_channels, 
                out_channels=out_channels, 
                kernel_size=self.k, 
                padding=self.p, 
                stride=self.s, 
                dilation=self.d
            ))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(
                kernel_size=self.pk, 
                stride=self.ps, 
                padding=self.pp, 
                dilation=self.pd
            ))
            current_in_channels = out_channels
        
        layers.append(nn.Flatten())
        
        # Group sequence into single executable block
        self.cnn_blocks = nn.Sequential(*layers)
        self.linear = nn.Linear(flattened_size, output_dim)

    def _check_algebra(self, length):
        ''' 
        Verifies the tensor internal and external dimensions before building the network. 

        Parameters:
        - length: The initial length of the input waveform sequence.

        Returns:
        - length: The final length of the sequence after all sequential operations.
        '''

        for i, _ in enumerate(self.channels):
            # Conv1d array length
            length = math.floor((length + 2 * self.p - self.d * (self.k - 1) - 1) / self.s + 1)
            if length <= 0:
                raise ValueError(f"\n[ALGEBRA ERROR] Conv1d Layer {i+1} reduced length to {length}.")
            
            # MaxPool1d array length
            length = math.floor((length + 2 * self.pp - self.pd * (self.pk - 1) - 1) / self.ps + 1)
            if length <= 0:
                raise ValueError(f"\n[ALGEBRA ERROR] MaxPool1d Layer {i+1} reduced length to {length}.")
                
        return length
        
    def forward(self, x):
        """ 
        Passes the waveform tensor through the network layers. 
        
        Parameters:
        - x: Input tensor of shape [Batch, in_channels, sequence_length]
        
        Returns:
        - features: Output tensor of shape [Batch, output_dim]
        """
        x = self.cnn_blocks(x)
        features = self.linear(x)
        
        return features

class ExtrinsicMLP(nn.Module):
    def __init__(self, mlp_config):
        '''
        Initialises a Multi-Layer Perceptron to compress raw extrinsic physical parameters 
        into an embedding vector.
        
        Parameters:
        - mlp_config: A dictionary containing
            - 'in_features': Number of input features (default 5).
            - 'hidden_features': List of hidden layer dimensions.
            - 'out_features': Number of output features.
        '''
        super(ExtrinsicMLP, self).__init__()
        
        in_features = mlp_config.get('in_features', 5)
        hidden_features = mlp_config.get('hidden_features', [16, 32])
        out_features = mlp_config.get('out_features', 32)
        
        layers = []
        current_in = in_features
        for h_dim in hidden_features:
            layers.append(nn.Linear(current_in, h_dim))
            layers.append(nn.ReLU())
            current_in = h_dim
            
        layers.append(nn.Linear(current_in, out_features))
        
        self.mlp_blocks = nn.Sequential(*layers)
        
    def forward(self, x):
        """ 
        Passes the extrinsic parameters through the network layers. 
        
        Parameters:
        - x: Input tensor of shape [Batch, in_features]
        
        Returns:
        - features: Output tensor of shape [Batch, out_features]
        """
        return self.mlp_blocks(x)


''' QUICK TEST '''
if __name__ == "__main__":
    test_length = 256
    test_config = {
        'in_channels': 1, 'channels': [16, 32, 64],
        'kernel_size': 9, 'padding': 4, 'stride': 1, 'dilation': 1,
        'pool_size': 2, 'pool_stride': 2, 'pool_padding': 0, 'pool_dilation': 1
    }
    dummy_batch = torch.randn(32, 1, test_length)
    embedder = WaveformEmbedding(sequence_length=test_length, embedding_config=test_config, output_dim=64)
    print(f"CNN output shape: {embedder(dummy_batch).shape}")
    print("Architectural CNN algebra mathematically consistent.")

    test_mlp_config = {'in_features': 5, 'hidden_features': [16, 32], 'out_features': 32}
    dummy_extrinsics = torch.randn(32, 5)
    mlp = ExtrinsicMLP(mlp_config=test_mlp_config)
    print(f"MLP Output shape: {mlp(dummy_extrinsics).shape}")
    