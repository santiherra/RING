import torch
import numpy as np

def get_gpu_noise_and_whiten(hp, hx, F_plus, F_cross, time_delays, asd_tensors, scale, sample_rate, duration, device):
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
    n_det = F_plus.shape[0]
    df = 1.0 / duration

    scale = float(scale)
    scale_factor = 1.0 / scale

    hp_scaled = hp * scale_factor
    hx_scaled = hx * scale_factor

    freqs = torch.fft.rfftfreq(seq_len, d=1.0 / sample_rate, device=device)
    whitened_channels = []
    snr_sq_total = torch.zeros(batch_size, device=device)

    for i in range(n_det):
        h_det = (F_plus[i] * hp_scaled) + (F_cross[i] * hx_scaled)
        
        # Time delay in frequency domain
        phase_shift = torch.exp(-1j * 2.0 * np.pi * freqs * time_delays[i])
        H_k = torch.fft.rfft(h_det) * phase_shift

        # Compute detector-specific matched-filter SNR
        h_fd = H_k * (1.0 / sample_rate)
        asd_scaled = asd_tensors[i] * scale_factor
        snr_sq_det = 4.0 * torch.sum((torch.abs(h_fd)**2) / (asd_scaled**2 + 1e-30), dim=1) * df
        snr_sq_total += snr_sq_det

        # Frequency-domain whitening
        W_k = np.sqrt(2.0 / sample_rate) / (asd_scaled + 1e-30)
        whitened_H_k = H_k * W_k
        whitened_td = torch.fft.irfft(whitened_H_k, n=seq_len)

        # Independent time-domain white noise realization per detector
        noise_td = torch.randn(batch_size, seq_len, device=device)
        d_white = whitened_td + noise_td
        whitened_channels.append(d_white)

    network_input = torch.stack(whitened_channels, dim=1)
    snr_net = torch.sqrt(snr_sq_total)

    return network_input, snr_net

if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 1024
    n_det = 2
    sample_rate = 4096.0
    duration = seq_len / sample_rate
    scale = 1.0

    hp = torch.randn(batch_size, seq_len)
    hx = torch.randn(batch_size, seq_len)
    F_plus = torch.tensor([1.0, 0.5])
    F_cross = torch.tensor([0.5, 1.0])
    time_delays = torch.tensor([0.01, -0.01])
    asd_tensors = torch.ones(n_det, seq_len // 2 + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp, hx, F_plus, F_cross, time_delays, asd_tensors = [x.to(device) for x in [hp, hx, F_plus, F_cross, time_delays, asd_tensors]]

    network_input, snr_net = get_gpu_noise_and_whiten(hp, hx, F_plus, F_cross, time_delays, asd_tensors, scale, sample_rate, duration, device)
    
    print("Network input shape:", network_input.shape)
    print("SNR shape:", snr_net.shape)
