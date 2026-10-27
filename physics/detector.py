import numpy as np
import bilby

class VectorisedDetectorProjector:
    def __init__(self, detector_names, sample_rate, duration):
        '''
        Vectorises Bilby interferometers and static ASDs to avoid CPU/GPU bottlenecks 
        during training loops.

        Parameters:
        - detector_names: List of detector names
        - sample_rate: Sampling rate
        - duration: Duration of the signal (seconds)
        '''
        self.detector_names = detector_names
        self.interferometers = []
        self.asd_tensors = []
        
        seq_len = int(duration * sample_rate)
        freqs = np.fft.rfftfreq(seq_len, d=1.0 / sample_rate)

        for name in detector_names:
            det = bilby.gw.detector.get_empty_interferometer(name)
            self.interferometers.append(det)
            
            if name in ['H1', 'L1']:
                psd_obj = bilby.gw.detector.PowerSpectralDensity.from_aligo()
            elif name == 'V1':
                psd_obj = bilby.gw.detector.PowerSpectralDensity.from_advanced_virgo()
            else:
                psd_obj = bilby.gw.detector.PowerSpectralDensity.from_aligo()

            psd_arr = psd_obj.power_spectral_density_interpolated(freqs)
            psd_arr[np.isinf(psd_arr)] = 1e-46
            asd_arr = np.sqrt(psd_arr)
            self.asd_tensors.append(asd_arr)
            
        self.asd_tensors = np.array(self.asd_tensors, dtype=np.float32)

    def project_batch(self, ra, dec, psi, gps_time):
        '''
        Vectorised computation of antenna patterns and time delays for a batch.
        
        Parameters:
        - ra: Right ascension (radians)
        - dec: Declination angle (radians)
        - psi: Polarisation angle (radians)
        - gps_time: GPS time (seconds)

        Returns:
        - F_plus: Plus-polarisation antenna function (shape: [n_detectors, batch_size])
        - F_cross: Cross-polarisation antenna function (shape: [n_detectors, batch_size])
        - time_delays: Time delays (shape: [n_detectors, batch_size])
        - asd_tensors: ASD tensors (shape: [n_detectors, n_frequencies])
        '''
        batch_size = len(ra)
        n_det = len(self.detector_names)
        
        F_plus = np.zeros((n_det, batch_size), dtype=np.float32)
        F_cross = np.zeros((n_det, batch_size), dtype=np.float32)
        time_delays = np.zeros((n_det, batch_size), dtype=np.float32)

        for i, det in enumerate(self.interferometers):
            for j in range(batch_size):
                F_plus[i, j] = det.antenna_response(ra[j], dec[j], gps_time[j], psi[j], 'plus')
                F_cross[i, j] = det.antenna_response(ra[j], dec[j], gps_time[j], psi[j], 'cross')
                time_delays[i, j] = det.time_delay_from_geocenter(ra[j], dec[j], gps_time[j])

        return F_plus, F_cross, time_delays, self.asd_tensors

''' QUICK TEST '''
if __name__ == "__main__":
    detectorprojector = VectorisedDetectorProjector(detector_names=["H1", "L1"], sample_rate=4096, duration=0.0625)
    proj_batch = detectorprojector.project_batch(
        ra=np.array([1.5, 2.0]),
        dec=np.array([-0.5, 0.0]),
        psi=np.array([0.8, 1.0]),
        gps_time=np.array([1420950000.0, 1420950001.0])
    )
    print(f"F_plus shape: {proj_batch[0].shape}, F_cross shape: {proj_batch[1].shape}, time_delays shape: {proj_batch[2].shape}, asd_tensors shape: {proj_batch[3].shape}")
    