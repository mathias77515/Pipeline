import numpy as np
import pickle
from mapmaking.systematics import give_cl_cmb, arcmin2rad

import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
import matplotlib.pyplot as plt
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

class PipelineExternalData:

    def __init__(self, external, nside, skyconfig):
        
        self.external = external
        self.nside = nside
        self.skyconfig = skyconfig


    def read_pkl(self, name):

        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def _get_cmb(self, seed, r=0, Alens=1):
        
        mycls = give_cl_cmb(r=r, Alens=Alens)

        np.random.seed(seed)
        cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T
        return cmb

    def _get_ave_map(self, central_nu, bw, model, nb=10):

        is_cmb = False
        model = []
        for key in self.skyconfig.keys():
            if key == 'cmb':
                is_cmb = True
            else:
                model += [self.skyconfig[key]]



        sky = pysm3.Sky(nside=self.nside, preset_strings=model, output_unit="uK_CMB")
        edges_min = central_nu - bw/2
        edges_max = central_nu + bw/2
        bandpass_frequencies = np.linspace(edges_min, edges_max, nb) * u.GHz

        if is_cmb:
            cmb = self._get_cmb(self.skyconfig['cmb'])
            return np.array(sky.get_emission(bandpass_frequencies)).T + cmb
        else:
            return np.array(sky.get_emission(bandpass_frequencies)).T
    
    def _get_fwhm(self, nu):
        return self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'fwhm{nu:.0f}']
    
    def _get_noise(self, nu):

        sigma = self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'noise{nu:.0f}']
        out = np.random.standard_normal(np.ones((12*self.nside**2, 3)).shape) * sigma
        return out
    
    def run(self, fwhm=False, noise=True):

        self.maps = np.zeros((len(self.external), 12*self.nside**2, 3))

        for inu, nu in enumerate(self.external):
            self.maps[inu] = self._get_ave_map(nu, 10, model)
            if noise:
                self.maps[inu] += self._get_noise(nu)
            if fwhm:
                C = HealpixConvolutionGaussianOperator(fwhm=arcmin2rad(self._get_fwhm(nu)))
                self.maps[inu]= C(self.maps[inu])


        
        
    

