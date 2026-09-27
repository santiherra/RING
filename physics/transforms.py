import torch
import numpy as np

def get_gpu_noise_and_whiten(hp, hx, F_plus, F_cross, asd_tensor, scale, sample_rate, duration, device):
    '''
    Projects polarizations to detector, adds colored Gaussian noise, and whitens on GPU.

    Parameters:
    - hp: plus polarisation. Tensor of shape [batch_size, seq_len].
    - hx: cross polarisation. Tensor of shape [batch_size, seq_len].
    - F_plus: plus antenna pattern. Tensor of shape [batch_size, 1].
    - F_cross: cross antenna pattern. Tensor of shape [batch_size, 1].
    - asd_tensor: amplitude spectral density. Tensor of shape [batch_size, n_freqs].
    - scale: Scaling factor for the signal.
    - sample_rate: Sampling rate of the signal.
    - duration: Duration of the signal in seconds.

    Returns:
    - network input: Whitened and scaled noisy signal.Tensor of shape [batch_size, 1, seq_len].
    - snr: Optimal snr of the signal. Tensor of shape [batch_size]
    '''
    batch_size, seq_len = hp.shape
    df = 1.0 / duration

    # Projected waveform in the detector frame
    h_det = (F_plus * hp) + (F_cross * hx)

    scale = float(scale)
    scale_factor = 1.0 / scale
    h_det_scaled = h_det * scale_factor
    asd_scaled = asd_tensor * scale_factor

    h_det_fd = torch.fft.rfft(h_det) * (1.0 / sample_rate)
    snr_sq = 4.0 * torch.sum((torch.abs(h_det_fd)**2) / (asd_tensor**2 + 1e-30), dim=1) * df
    snr = torch.sqrt(snr_sq)

    H_k = torch.fft.rfft(h_det_scaled)
    h_fd = H_k * (1.0 / sample_rate)
    snr_sq = 4.0 * torch.sum((torch.abs(h_fd)**2) / (asd_scaled**2 + 1e-30), dim=1) * df
    snr = torch.sqrt(snr_sq)

    W_k = 1.0 / (asd_scaled * np.sqrt(df) + 1e-30)
    whitened_H_k = H_k * W_k

    whitened_h_td = torch.fft.irfft(whitened_H_k, n=seq_len)

    white_noise_td = torch.randn(batch_size, seq_len, device=device)
    
    network_input = whitened_h_td + white_noise_td
    
    return network_input.unsqueeze(1), snr
