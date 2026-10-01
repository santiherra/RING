import numpy as np
import bilby

def get_detector_responses(detector_names, ra, dec, psi, gps_time, sample_rate, duration):
    '''
    Computes antenna patterns, geometric time delays, and ASD arrays for a list of detectors.

    Parameters:
    - detector_names: List of detector identifiers
    - ra, dec, psi: Extrinsic angular parameters, right ascension, declination and polarisation angle (radians)
    - gps_time: Geocenter GPS time (seconds)
    - sample_rate: Sampling frequency (Hz)
    - duration: Signal duration (seconds)

    Returns:
    - responses: Dict containing F_plus, F_cross, time_delay, and asd per detector.
    '''
    seq_len = int(duration * sample_rate)
    freqs = np.fft.rfftfreq(seq_len, d=1.0 / sample_rate)

    responses = {
        'names': detector_names,
        'F_plus': [],
        'F_cross': [],
        'time_delay': [],
        'asd': []
    }

    for name in detector_names:
        det = bilby.gw.detector.get_empty_interferometer(name)
        
        f_p = det.antenna_response(ra, dec, gps_time, psi, 'plus')
        f_c = det.antenna_response(ra, dec, gps_time, psi, 'cross')
        delay = det.time_delay_from_geocenter(ra, dec, gps_time)
        
        if name in ['H1', 'L1']:
            psd_obj = bilby.gw.detector.PowerSpectralDensity.from_aligo()
        elif name == 'V1':
            psd_obj = bilby.gw.detector.PowerSpectralDensity.from_advanced_virgo()
        else:
            psd_obj = bilby.gw.detector.PowerSpectralDensity.from_aligo()

        psd_arr = psd_obj.power_spectral_density_interpolated(freqs)
        psd_arr[np.isinf(psd_arr)] = 1e-46
        asd_arr = np.sqrt(psd_arr)

        responses['F_plus'].append(f_p)
        responses['F_cross'].append(f_c)
        responses['time_delay'].append(delay)
        responses['asd'].append(asd_arr)

    responses['F_plus'] = np.array(responses['F_plus'], dtype=np.float32)
    responses['F_cross'] = np.array(responses['F_cross'], dtype=np.float32)
    responses['time_delay'] = np.array(responses['time_delay'], dtype=np.float32)
    responses['asd'] = np.array(responses['asd'], dtype=np.float32)

    return responses