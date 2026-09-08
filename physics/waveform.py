import numpy as np
import qnm

# Conversion factor: Solar mass to seconds (G * M_sun / c^3)
M_SUN_S = 4.925491025543575e-6 

def generate_ringdown(Mf, af, 
                      x_p_0, y_p_0, x_c_0, y_c_0, 
                      x_p_1, y_p_1, x_c_1, y_c_1, 
                      sample_rate=4096, duration=0.05):
    """
    Generates the time-domain ringdown waveform h_+ and h_x 
    for the (2,2,0) and (2,2,1) modes using Cartesian amplitudes.
    
    Parameters:
    - Mf: Remnant mass in Solar Masses
    - af: Dimensionless remnant spin (0 to 1)
    - x_p_0, y_p_0, x_c_0, y_c_0: Cartesian amplitude components for fundamental mode (2,2,0)
    - x_p_1, y_p_1, x_c_1, y_c_1: Cartesian amplitude components for first overtone (2,2,1)
    - sample_rate: Detector sampling rate (default 4096)
    - duration: Length of the ringdown signal to generate in seconds (default 0.05s)

    Returns:
    - t: Time array (seconds)
    - hp: Plus polarisation strain array
    - hx: Cross polarisation strain array
    """
    
    t = np.arange(0, duration, 1.0 / sample_rate)
    t_dim = t / (Mf * M_SUN_S)
    
    # QNM: fundamental mode (2, 2, 0), first overtone (2, 2, 1)
    mode_0 = qnm.modes_cache(s=-2, l=2, m=2, n=0)
    mode_1 = qnm.modes_cache(s=-2, l=2, m=2, n=1)
    
    # Complex dimensionless frequencies
    omega_complex_0, _, _ = mode_0(a=af)
    omega_complex_1, _, _ = mode_1(a=af)
    
    omega_0 = np.real(omega_complex_0)
    gamma_0 = -np.imag(omega_complex_0)
    
    omega_1 = np.real(omega_complex_1)
    gamma_1 = -np.imag(omega_complex_1)
    
    # Polarisations, fundamental mode (n=0)
    hp_0 = np.exp(-gamma_0 * t_dim) * (x_p_0 * np.cos(omega_0 * t_dim) + y_p_0 * np.sin(omega_0 * t_dim))
    hc_0 = np.exp(-gamma_0 * t_dim) * (x_c_0 * np.cos(omega_0 * t_dim) + y_c_0 * np.sin(omega_0 * t_dim))

    # Polarisations, first overtone (n=1)
    hp_1 = np.exp(-gamma_1 * t_dim) * (x_p_1 * np.cos(omega_1 * t_dim) + y_p_1 * np.sin(omega_1 * t_dim))
    hc_1 = np.exp(-gamma_1 * t_dim) * (x_c_1 * np.cos(omega_1 * t_dim) + y_c_1 * np.sin(omega_1 * t_dim))
    
    hp = hp_0 + hp_1
    hx = hc_0 + hc_1
    
    return t, hp, hx


''' QUICK TEST '''

if __name__ == "__main__":
    # Example: GW250114
    t, hp, hx = generate_ringdown(
        Mf=65.0, af=0.7, 
        x_p_0=1.0, y_p_0=0.5, x_c_0=0.8, y_c_0=-0.2, 
        x_p_1=0.5, y_p_1=0.2, x_c_1=0.4, y_c_1=-0.1
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
    