import torch
import torch.nn as nn

class WaveformEmbedding(nn.Module):
    def __init__(self, in_channels=1, output_dim=64):
        """
        Initializes a 1D Convolutional Neural Network that compresses a time-domain ringdown
        into a dense vector of summary features.
        
        Parameters:
        - in_channels: Number of detectors (1 for just H1, 2 for H1 + L1).
        - output_dim: The number of summary features to output (default 64).
        """
        super(WaveformEmbedding, self).__init__()
        
        # Scan the raw waveform. Input: [Batch, 1, 256] -> Output: [Batch, 16, 128]
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Extract intermediate temporal features. Input: [Batch, 16, 128] -> Output: [Batch, 32, 64]
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding=4)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Extract deep abstract features. Input: [Batch, 32, 64] -> Output: [Batch, 64, 32]
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, padding=4)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        # After 3 max pooling layers, the 256 length is divided into 8 parts of length 32.
        # There are 64 channels: 64 * 32 = 2048 flattened features.
        # Map the 2048 features down to the strict output dimension (default 64).
        self.linear = nn.Linear(64 * 32, output_dim)
        
    def forward(self, x):
        """ 
        Passes the waveform tensor through the network layers 
        
        Parameters:
        - x: Input tensor of shape [Batch, in_channels, 256]
        
        Returns:
        - features: Output tensor of shape [Batch, output_dim]
        """

        # Pass the input through the convolutional layers with ReLU activations and max pooling
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        # Flatten the output and pass it through the linear layer to get the final summary features
        x = self.flatten(x)
        features = self.linear(x)
        
        return features

''' QUICK TEST '''
if __name__ == "__main__":
    # Simulate a batch of 32 waveforms (1 channel, 256 time steps)
    dummy_batch = torch.randn(32, 1, 256)
    
    embedder = WaveformEmbedding(in_channels=1, output_dim=64)
    summary_features = embedder(dummy_batch)
    
    print(f"Input shape: {dummy_batch.shape}")
    print(f"Output shape: {summary_features.shape}")
    print("CNN is mathematically complete.")
    