import numpy as np
import bilby

# Hanford detector object
H1 = bilby.gw.detector.get_empty_interferometer("H1")

# Uncomment the line below to add the Livingston detector.
# L1 = bilby.gw.detector.get_empty_interferometer("L1")

def project_to_detector(hp, hx, ra, dec, psi, gps_time, detector="H1"):
    """
    Calculates the exact antenna patterns F_+ and F_x, and projects 
    the radiation-frame ringdown polarizations into the observer frame.
    
    Parameters:
    - hp, hx: Time-series arrays (1D Numpy arrays from waveform.py)
    - ra, dec, psi: Right ascension, declination, and polarization angles (radians)
    - gps_time: The GPS time of the event (t_0)
    - detector: String ('H1' or 'L1')
    
    Returns:
    - h_det: The 1D observed strain array scaled for the chosen detector
    """
    
    # Detector geometry
    if detector == "H1":
        det = H1
    # Uncomment the lines below to add the Livingston detector.
    # elif detector == "L1": 
    #     det = L1
    else:
        raise ValueError(f"Detector {detector} not yet implemented.")
        
    # Antenna Patterns
    F_plus = det.antenna_response(ra, dec, gps_time, psi, 'plus')
    F_cross = det.antenna_response(ra, dec, gps_time, psi, 'cross')
    
    # Projection of the waveform components into the detector arm
    h_det = (F_plus * hp) + (F_cross * hx)
    
    return h_det


''' QUICK TEST '''

if __name__ == "__main__":
    # Test waveform 
    dummy_t = np.linspace(0, 0.05, 2048)
    dummy_hp = np.cos(2 * np.pi * 250 * dummy_t) * np.exp(-dummy_t / 0.01)
    dummy_hx = np.sin(2 * np.pi * 250 * dummy_t) * np.exp(-dummy_t / 0.01)
    
    # Example values
    test_gps_time = 1420950000.0  
    test_ra = 1.5
    test_dec = -0.5
    test_psi = 0.8
    
    # Projection
    h_observed = project_to_detector(
        dummy_hp, dummy_hx, 
        ra=test_ra, dec=test_dec, psi=test_psi, gps_time=test_gps_time, 
        detector="H1"
    )
    
    print(f"Raw H_plus max amplitude:  {np.max(abs(dummy_hp)):.4f}")
    print(f"Observed H1 max amplitude: {np.max(abs(h_observed)):.4f}")
    print("Notice how the detector's geometry scales down the raw signal amplitude!")
    