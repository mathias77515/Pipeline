import numpy as np
import healpy as hp

### QUBIC packages
import qubic




class Noise:

    def __init__(self, nside):
        
        self.nside = nside
        self.depth150 = 4.2   # muK.arcmin
        self.depth220 = 4.0   # muK.arcmin
        self.resol = hp.nside2resol(self.nside, arcmin=True)

    def _get_sigma_from_depths(self, depth):
        return depth / self.resol





noise = Noise(256)
print(noise._get_sigma_from_depths(noise.depth150))
print(noise._get_sigma_from_depths(noise.depth220))




































#