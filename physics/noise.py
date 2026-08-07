import numpy as np
import bilby

def generate_noise(duration, sample_rate, psd=None):
    """
    Generates a 1D time-domain array of colored Gaussian noise.

    Parameters:
    - duration: Length of the noise signal to generate in seconds
    - sample_rate: Detector sampling rate
    - psd: Optional bilby PowerSpectralDensity object. If None, defaults to ALIGO design sensitivity.

    Returns:
    - noise_td: 1D Numpy array of time-domain noise samples
    """

    # Calculate the number of samples based on duration and sample rate
    num_samples = int(duration * sample_rate)
    
    if psd is None:
        psd = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        
    # Unpack the tuple. Bilby returns (noise_data, frequencies)
    noise_fd, _ = psd.get_noise_realisation(sample_rate, duration)
    
    # Transform to the time domain (This yields a 1D array of exactly 256 points)
    noise_td = np.fft.irfft(noise_fd, n=num_samples) * sample_rate
    
    return noise_td


''' QUICK TEST '''

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import os
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from physics.waveform import generate_ringdown
    from physics.detector import project_to_detector

    test_duration = 0.0625
    test_sample_rate = 4096
    
    # Initialize PSD once to prevent terminal spam during tests
    import logging
    logging.getLogger('bilby').setLevel(logging.WARNING)
    test_psd = bilby.gw.detector.PowerSpectralDensity.from_aligo()
    
    # Generate the noise
    noise = generate_noise(duration=test_duration, sample_rate=test_sample_rate, psd=test_psd)
    
    # Generate a physical waveform (Scaled to a realistic 5e-21 strain)
    physical_scale = 5e-21
    t, hp, hx = generate_ringdown(
            Mf=65.0, af=0.7, 
            C_re_0=1.0 * physical_scale, C_im_0=0.5 * physical_scale, 
            C_re_1=0.8 * physical_scale, C_im_1=-0.2 * physical_scale, 
            iota=0.5, phi=1.2,
            duration=test_duration,
            sample_rate=test_sample_rate
        )

    # Project it to a detector
    h_observed = project_to_detector(
            hp, hx, 
            ra=1.5, dec=-0.5, psi=0.8, gps_time=1420950000.0, 
            detector="H1"
        )
    
    # Add them together to create the raw detector strain
    raw_signal = h_observed + noise

    # Plot the exact physical representation
    plt.figure(figsize=(10, 4))
    
    # Plot the raw signal and the physical ringdown waveform
    plt.plot(t, raw_signal, color='lightgray', label='Raw Signal + Noise', zorder=0)
    plt.plot(t, h_observed, color='orange', label='Raw Ringdown', zorder=5)
    
    plt.title('Physical LIGO Data (Strain ~ 1e-21)')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    