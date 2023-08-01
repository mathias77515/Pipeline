import qubic
import mapmaking.frequency_acquisition as frequency_acquisition
import numpy as np

class QubicNoise:
    
    def __init__(self, band, npointings, comm=None, size=1, detector_nep=4.7e-17):
        
        if band != 150 and band != 220:
            raise TypeError('Please choose the QubicWideBandNoise method.')
        
        dictfilename = 'dicts/pipeline_demo.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        
        d['TemperatureAtmosphere150']=None
        d['TemperatureAtmosphere220']=None
        d['EmissivityAtmosphere150']=None
        d['EmissivityAtmosphere220']=None
        d['detector_nep'] = detector_nep

        d['npointings'] = npointings
        d['comm'] = comm
        d['nprocs_instrument'] = size
        d['nprocs_sampling'] = 1
        
        self.dict = d.copy()
        self.dict['filter_nu'] = int(band)
        self.dict['nf_sub'] = 1
        self.dict['nf_recon'] = 1
        self.dict['type_instrument']=''
        self.acq = frequency_acquisition.QubicIntegrated(self.dict, Nsub=1, Nrec=1)
        
    def get_noise(self, det_noise, pho_noise):
        n = self.detector_noise() * 0
        
        if det_noise:
            n += self.detector_noise()
        if pho_noise:
            n += self.photon_noise()
        return n
    
    def photon_noise(self):
        return self.acq.get_noise(det_noise=False, photon_noise=True)
    
    def detector_noise(self):
        return self.acq.get_noise(det_noise=True, photon_noise=False)
    
    def total_noise(self, wdet, wpho):
        ndet = wdet * self.detector_noise()
        npho = wpho * self.photon_noise()
        return ndet + npho
    
    
class QubicWideBandNoise:
    
    def __init__(self, d, npointings, detector_nep=4.7e-17):
    
        self.d = d
        self.npointings = npointings
        self.detector_nep = detector_nep
        
        
    def total_noise(self, wdet, wpho150, wpho220):
        
        Qubic150 = QubicNoise(150, self.npointings, comm=self.d['comm'], size=self.d['nprocs_instrument'], detector_nep=self.detector_nep)
        Qubic220 = QubicNoise(220, self.npointings, comm=self.d['comm'], size=self.d['nprocs_instrument'], detector_nep=self.detector_nep)
        
        ndet = wdet * Qubic150.detector_noise()
        npho150 = wpho150 * Qubic150.photon_noise()
        npho220 = wpho220 * Qubic220.photon_noise()
        
        return ndet + npho150 + npho220
        

class QubicDualBandNoise:

    def __init__(self, d, npointings, detector_nep=4.7e-17):

        self.d = d
        self.npointings = npointings
        self.detector_nep = detector_nep

    def total_noise(self, wdet, wpho150, wpho220):
        
        Qubic150 = QubicNoise(150, self.npointings, comm=self.d['comm'], size=self.d['nprocs_instrument'], detector_nep=self.detector_nep)
        Qubic220 = QubicNoise(220, self.npointings, comm=self.d['comm'], size=self.d['nprocs_instrument'], detector_nep=self.detector_nep)
        
        ndet = wdet * Qubic150.detector_noise().ravel()
        npho150 = wpho150 * Qubic150.photon_noise().ravel()
        npho220 = wpho220 * Qubic220.photon_noise().ravel()
        
        return np.r_[ndet + npho150, ndet + npho220]

