import numpy as np
import qnm

# Conversion factor: Solar Mass to Seconds (G * M_sun / c^3)
M_SUN_S = 4.925491025543575e-6 

def spin_weighted_spherical_harmonic(iota, phi, l=2, m=2):
    """
    Calculates the s = -2 spin-weighted spherical harmonic (Approximating spheroidal as spherical for the 
    baseline network architecture).

    Parameters:
    - iota: Inclination angle (radians)
    - phi: Azimuthal angle (radians)
    - l: Spherical harmonic degree (default 2)
    - m: Spherical harmonic projection (default 2)

    Returns:
    - Y_lm: Complex value of the spin-weighted spherical harmonic at the given angles.
    """

    # Amplitude for s=-2, l=2, m=2
    amplitude = np.sqrt(5.0 / (64.0 * np.pi)) * (1.0 + np.cos(iota))**2
    
    # Complex angular dependence
    return amplitude * np.exp(2j * phi)

def generate_ringdown(Mf, af, C_re_0, C_im_0, C_re_1, C_im_1, 
                      iota, phi, sample_rate=4096, duration=0.05):
    """
    Generates the time-domain ringdown waveform h_+ and h_x 
    for the (2,2,0) and (2,2,1) modes using Cartesian amplitudes.
    
    Parameters:
    - Mf: Remnant mass in Solar Masses
    - af: Dimensionless remnant spin (0 to 1)
    - C_re_0, C_im_0: Cartesian amplitude components for fundamental mode (2,2,0)
    - C_re_1, C_im_1: Cartesian amplitude components for first overtone (2,2,1)
    - iota, phi: Fixed inclination and azimuthal angle of the binary
    - sample_rate: Detector sampling rate (default 4096)
    - duration: Length of the ringdown signal to generate in seconds (default 0.05s)

    Returns:
    - t: Time array (seconds)
    - hp: Plus polarization strain array
    - hx: Cross polarization strain array
    """
    
    # Create the physical time array (in seconds). Network default: t = 0 at the start of the ringdown
    t = np.arange(0, duration, 1.0 / sample_rate)
    
    # Dimensionless time
    t_dim = t / (Mf * M_SUN_S)
    
    # Fetch complex frequencies for the QNM (fundamental mode n = 0, and overtone n= 1)
    mode_0 = qnm.modes_cache(s=-2, l=2, m=2, n=0)
    mode_1 = qnm.modes_cache(s=-2, l=2, m=2, n=1)
    
    # Fetch omega using the exact keyword argument 'a' required by KerrSpinSeq
    omega_0, _, _ = mode_0(a=af)
    omega_1, _, _ = mode_1(a=af)
    
    # Construct the complex amplitudes from Cartesian inputs
    A_0 = C_re_0 + 1j * C_im_0
    A_1 = C_re_1 + 1j * C_im_1
    
    # Build the temporal part of the waveform: sum_n A_n * exp(-i * omega_n * t_dim)
    strain_time = (A_0 * np.exp(-1j * omega_0 * t_dim) + 
                   A_1 * np.exp(-1j * omega_1 * t_dim))
    
    # Apply the angular dependence
    Y_22 = spin_weighted_spherical_harmonic(iota, phi)
    
    # Full complex strain: h_+ - i h_x
    h_complex = strain_time * Y_22
    
    # Separate into plus and cross polarizations
    hp = np.real(h_complex)
    hx = -np.imag(h_complex)
    
    return t, hp, hx


''' QUICK TEST '''

if __name__ == "__main__":
    # Example using roughly GW250114-like parameters
    t, hp, hx = generate_ringdown(
        Mf=65.0, af=0.7, 
        C_re_0=1.0, C_im_0=0.5, 
        C_re_1=0.8, C_im_1=-0.2, 
        iota=0.5, phi=1.2
    )
    print(f"Generated {len(t)} data points.")
    print(f"h_plus preview: {hp[:5]}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(t, hp, label='h_plus')
    plt.plot(t, hx, label='h_cross')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.title('Ringdown Waveform')
    plt.legend()
    plt.grid()
    plt.show()
    