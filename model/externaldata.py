import numpy as np
import pickle
import yaml
#from mapmaking.systematics import give_cl_cmb, arcmin2rad
import mapmaking.systematics as acq

import pysm3

import pysm3.units as u
from pysm3 import utils
import healpy as hp
import matplotlib.pyplot as plt
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

class PipelineExternalData:

    def __init__(self, file, noise_only=False):
        
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        with open('noise.yml', "r") as stream:
            self.noise = yaml.safe_load(stream)
        
        self.noise_only = noise_only
        if self.noise_only:
            self.factor = 0
        else:
            self.factor = 1
        self.external_nus = self._read_external_nus()
        #print('external nus : ', self.external_nus)
        
        self.nside = self.params['Sky']['nside']
        self.skyconfig = self._get_sky_config()
        self.file = file
        
    
    def _get_sky_config(self):
        
        sky = {}
        for ii, i in enumerate(self.params['Sky'].keys()):
            #print(ii, i)

            if i == 'CMB':
                if self.params['Sky']['CMB']['cmb'][0]:
                    if self.params['Sky']['CMB']['cmb'][1] == 0:
                        seed = np.random.randint(10000000)
                        
                    else:
                        seed = self.params['Sky']['CMB']['cmb'][1]
                    #stop
                    sky['cmb'] = seed
                
            else:
                for jj, j in enumerate(self.params['Sky']['Foregrounds']):
                    #print(j, self.params['Foregrounds'][j])
                    if j == 'Dust':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['dust'] = 'd0'
                    elif j == 'Synchrotron':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['synchrotron'] = 's0'

        return sky
    def _get_depth(self, nus):
    
        res = []
    
        for mynu in nus:
            
            
            is_bicep = np.sum(mynu == np.array([95, 150, 220])) != 0
            is_planck = np.sum(mynu == np.array([30, 44, 70, 100, 143, 217, 353])) != 0

            if is_bicep:
                
                index = self.noise['Bicep']['frequency'].index(mynu) if mynu in self.noise['Bicep']['frequency'] else -1
                if index != -1:
                    d = self.noise['Bicep']['depth_p'][index]
                    res.append(d)
                else:
                    res.append(None)  # Fréquence non trouvée, ajout d'une valeur None
    
            elif is_planck:
                index = self.noise['Planck']['frequency'].index(mynu) if mynu in self.noise['Planck']['frequency'] else -1
                if index != -1:
                    d = self.noise['Planck']['depth_p'][index]
                    res.append(d)
                else:
                    res.append(None)  # Fréquence non trouvée, ajout d'une valeur None
    
        return res
    def _update_data(self, maps, nus):

        data = self.read_pkl(self.file)

        data['maps_ext'] = maps
        data['nus_ext'] = nus

        with open(self.file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def _read_external_nus(self):

        allnus_pl = [30, 44, 70, 100, 143, 217, 353]
        allnus = []

        if self.params['Data']['planck']:
            allnus += allnus_pl
        #allnus = allnus_bk + allnus_pl
        nus = []

        for inu, nu in enumerate(allnus):
            if inu < 3:
                nus += [allnus[inu]]
            else:
                 nus += [allnus[inu]]
        #print(nus)
        #stop
        return nus
    def read_pkl(self, name):
        
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data
    def _get_cmb(self, seed, r=0, Alens=1):
        
        mycls = acq.give_cl_cmb(r=r, Alens=Alens)

        np.random.seed(seed)
        cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T
        return cmb
    def _get_ave_map(self, central_nu, bw, nb=100):

        is_cmb = False
        model = []
        for key in self.skyconfig.keys():
            if key == 'cmb':
                is_cmb = True
            else:
                model += [self.skyconfig[key]]

        mysky = np.zeros((12*self.params['Sky']['nside']**2, 3))

        if len(model) != 0:
            sky = pysm3.Sky(nside=self.nside, preset_strings=model)
            edges_min = central_nu - bw/2
            edges_max = central_nu + bw/2
            bandpass_frequencies = np.linspace(edges_min, edges_max, nb)
            print(f'Integrating bandpass from {edges_min} GHz to {edges_max} GHz with {nb} frequencies.')
            mysky += np.array(sky.get_emission(bandpass_frequencies * u.GHz, None) * utils.bandpass_unit_conversion(bandpass_frequencies * u.GHz, None, u.uK_CMB)).T / 1.5


        if is_cmb:
            cmb = self._get_cmb(self.skyconfig['cmb'])
            mysky += cmb
            
        return mysky * self.factor   
    def _get_fwhm(self, nu):
        fwhmi = self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'fwhm{nu:.0f}']
        return fwhmi
    def _get_noise(self, nu):
        
        np.random.seed(None)
        #print(nu)
        sigma = self._get_depth([nu])[0] / hp.nside2resol(self.nside, arcmin=True)
        #print(sigma)
        #sigma = np.array([hp.ud_grade(self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'noise{nu:.0f}'][:, i], self.params['Sky']['nside']) for i in range(3)]).T
        out = np.random.standard_normal(np.ones((12*self.params['Sky']['nside']**2, 3)).shape) * sigma
        return out
    def run(self, fwhm=False, noise=True):

        '''

        Method that create global variables such as :

            - self.maps : Frequency maps from external data with shape (Nf, Npix, Nstk)
            - self.external_nus  : Frequency array [GHz]

        '''

        self.maps = np.zeros((len(self.external_nus), 12*self.nside**2, 3))
        self.fwhm_ext = []
        for inu, nu in enumerate(self.external_nus):
            #print(self.external_nus, inu, nu)
            self.maps[inu] = self._get_ave_map(nu, nu*self.params['bandwidth_planck'], nb=self.params['nb_integration'])
            if noise:
                self.maps[inu] += self._get_noise(nu)
            if fwhm:
                C = HealpixConvolutionGaussianOperator(fwhm=acq.arcmin2rad(self._get_fwhm(nu)))
                self.fwhm_ext.append(acq.arcmin2rad(self._get_fwhm(nu)))
                self.maps[inu] = C(self.maps[inu])
            else:
                self.fwhm_ext.append(0)

        #self._update_data(self.maps, self.external_nus)
        
        #with open(self.params['Data']['datafilename'], 'rb') as f:
        #    data = pickle.load(f)
        #self.maps = np.concatenate((data['maps'], self.maps), axis=0)
        #self.nus = np.concatenate((data['nus'], self.external), axis=0)
        #self.maps[:, ~self.seenpix, :] = 0
        #self.maps[:, :, 0] = 0
        #if self.rank == 0:
        #    self.save_pkl(self.params['Data']['datafilename'], {'maps':self.maps, 'nus':self.nus})


        
        
    

