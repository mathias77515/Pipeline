#### General packages
import pickle
import os
import sys
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import scipy
import healpy as hp
import emcee
import yaml
from multiprocessing import Pool
import time

sys.path.append('/pbs/home/t/tlaclave/sps/Pipeline')
 
#### QUBIC packages
import qubic
from qubic import NamasterLib as nam
from qubicpack.utilities import Qubic_DataDir
from qubic import QubicSkySim as qss
from pysimulators import FitsArray
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
import mapmaking.systematics as acq
from qubic import mcmc
from qubic import AnalysisMC as amc
from qubic.beams import BeamGaussian
import fgb.component_model as c
import fgb.mixing_matrix as mm
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
#from pipeline import *
from pyoperators import *
#from preset.preset import *

class Spectra:
    '''
    Class to compute the different spectra for our realisations
    '''

    def __init__(self, iter):

        self.iter = iter
        with open('spectrum_config.yml', "r") as stream:
            self.param_spectrum = yaml.safe_load(stream)
        self.pipeline = self.param_spectrum['data']['pipeline']
        self.path_sky = self.param_spectrum['data']['path_sky']
        self.path_noise = self.param_spectrum['data']['path_noise']

        if self.pipeline == 'CMM':
            self.pkl_sky = self.find_data(self.path_sky)
            self.pkl_noise = self.find_data(self.path_noise)
            self.components_map = self.pkl_sky['components']
            self.noise_map = self.pkl_noise['components']
            self.ncomp = np.shape(self.components_map)[0]
            self.nsub = self.data_parameters['QUBIC']['nsub']
            self.allfwhm = self.sims.joint.qubic.allfwhm

        if self.pipeline == 'FMM':
            self.pkl_sky = self.find_data(self.path_sky)
            self.pkl_noise = self.find_data(self.path_noise)
            self.sky_maps = self.pkl_sky['maps']
            self.maps_parameters = self.pkl_sky['parameters']
            self.noise_maps = self.pkl_noise['maps']
            self.nrec = self.param_spectrum['simu']['nrec']
            self.nsub = self.pkl_sky['parameters']['QUBIC']['nsub']
            self.nside = self.pkl_sky['parameters']['Sky']['nside']
            self.fsub = int(self.nsub / self.nrec)
            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.my_dict, _ = self.get_dict()

        self.ell, self.namaster = NamasterEll(self.iter).ell()
        
        if self.maps_parameters['QUBIC']['convolution'] is True:
            _, allnus150, _, _, _, _ = qubic.compute_freq(150, Nfreq=int(self.nsub/2)-1, relative_bandwidth=0.25)
            _, allnus220, _, _, _, _ = qubic.compute_freq(220, Nfreq=int(self.nsub/2)-1, relative_bandwidth=0.25)
            self.allnus = np.array(list(allnus150) + list(allnus220))
            fwhm = self.allfwhm()
            allfwhm = []
            for i in range(self.nrec):
                allfwhm.append(fwhm[(i+1)*self.fsub - 1])
            self.allfwhm = fwhm
        else:
            self.allfwhm = np.zeros(self.nrec)

    def find_data(self, path):
        '''
        Function to extract the pickle file of one realisation associated to the path and the iteration number given.

        Argument :
            - path (str) : path where the pkl files are located

        Return :
            - pkl file dictionary (dict)
        '''

        data_names = os.listdir(path)
        one_realisation = pickle.load(open(path + '/' + data_names[self.iter], 'rb'))

        return one_realisation

    def get_dict(self):
        """
        Method to modify the qubic dictionary.
        """

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()
        params = self.pkl_sky['parameters']

        args = {'npointings':params['QUBIC']['npointings'], 
                'nf_recon':params['QUBIC']['nrec'], 
                'nf_sub':params['QUBIC']['nsub'], 
                'nside':params['Sky']['nside'], 
                'MultiBand':True, 
                'period':1, 
                'RA_center':params['QUBIC']['RA_center'], 
                'DEC_center':params['QUBIC']['DEC_center'],
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'comm':self.comm, 
                'dtheta':params['QUBIC']['dtheta'],
                'nprocs_sampling':1, 
                'nprocs_instrument':self.size,
                'photon_noise':True, 
                'nhwp_angles':params['QUBIC']['nhwp_angles'], 
                'effective_duration':3, 
                'filter_relative_bandwidth':delta_nu_over_nu, 
                'type_instrument':'wide', 
                'TemperatureAtmosphere150':None, 
                'TemperatureAtmosphere220':None,
                'EmissivityAtmosphere150':None, 
                'EmissivityAtmosphere220':None, 
                'detector_nep':float(params['QUBIC']['detector_nep']), 
                'synthbeam_kmax':params['QUBIC']['synthbeam_kmax']}

        args_mono = args.copy()
        args_mono['nf_recon'] = 1
        args_mono['nf_sub'] = 1

        ### Get the default dictionary
        dictfilename = 'dicts/pipeline_demo.dict'
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        dmono = d.copy()

        for i in args.keys():
            d[str(i)] = args[i]
            dmono[str(i)] = args_mono[i]

        return d, dmono

    def get_ultrawideband_config(self):
        """
        Method that pre-compute UWB configuration.
        """

        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
        return nu_ave, 2*delta/nu_ave
    
    def synthbeam(self, synthbeam_peak150_fwhm, dtype=np.float32):
        sb = SyntheticBeam()
        sb.dtype = np.dtype(dtype)
        nripples = self.my_dict['nripples']
        synthbeam_peak150_fwhm = np.radians(self.my_dict['synthbeam_peak150_fwhm'])
        if not nripples:
            sb.peak150 = BeamGaussian(synthbeam_peak150_fwhm)
        else:
            sb.peak150 = BeamGaussianRippled(synthbeam_peak150_fwhm,
                                             nripples=nripples)
        return sb
    
    def allfwhm(self):
        '''
        Function to compute the fwhm for all sub bands.

        Return :
            - allfwhm (list [nrec * nsub])
        '''

        synthbeam_peak150_fwhm = np.radians(self.my_dict['synthbeam_peak150_fwhm'])
        synthbeam = self.synthbeam(synthbeam_peak150_fwhm, dtype=np.float32)
        allfwhm = synthbeam.peak150.fwhm * (150 / self.allnus)

        return allfwhm

    def compute_auto_spectrum(self, map, fwhm):
        '''
        Function to compute the auto-spectrum of a given map

        Argument : 
            - map(array) [nrec/ncomp, npix, nstokes] : map to compute the auto-spectrum
            - allfwhm(float) : in radian
        Return : 
            - (list) [len(ell)] : BB auto-spectrum
        '''

        DlBB = self.namaster.get_spectra(map=map.T, map2=None, beam_correction = np.rad2deg(fwhm))[1][:, 2]
        return DlBB

    def compute_cross_spectrum(self, map1, map2):
        '''
        Function to compute cross-spectrum, taking into account the different resolution of each sub-bands

        Arguments :
            - map1 & map2 (array [nrec/ncomp, npix, nstokes]) : the two maps needed to compute the cross spectrum
            - fwhm1 & fwhm2 (float) : the respective fwhm for map1 & map2 in radian

        Return : 
            - (list) [len(ell)] : BB cross-spectrum
        '''

        # Put the map with the highest resolution at the worst one before doing the cross spectrum
        # Important because the two maps had to be at the same resolution and you can't increase the resolution
        #if fwhm1<fwhm2 :
            #C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm2**2 - fwhm1**2))
            #convoluted_map = map1
        return self.namaster.get_spectra(map=map1.T, map2=map2.T, beam_correction = 0)[1][:, 2]
        #else:
        #    C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm1**2 - fwhm2**2))
        #    convoluted_map = C*map2
        #    return self.namaster.get_spectra(map=map1.T, map2=convoluted_map.T, beam_correction = np.rad2deg(fwhm1))[1][:, 2]

    def compute_array_power_spectra(self, maps):
        ''' 
        Function to fill an array with all the power spectra computed

        Argument :
            - maps (array [nreal, nrec/ncomp, npix, nstokes]) : all your realisation maps

        Return :
            - power_spectra_array (array [nrec/ncomp, nrec/ncomp]) : element [i, i] is the auto-spectrum for the reconstructed sub-bands i 
                                                                     element [i, j] is the cross-spectrum between the reconstructed sub-band i & j
        '''

        if self.pipeline == 'FMM':
            idx_lenght = self.nrec
        elif self.pipeline == 'CMM':
            idx_lenght = self.ncomp
        power_spectra_array = np.zeros((idx_lenght, idx_lenght, len(self.ell)))

        for i in range(idx_lenght):
            for j in range(idx_lenght):
                if i==j :
                    # Compute the auto-spectrum
                    power_spectra_array[i,j] = self.compute_auto_spectrum(maps[i], self.allfwhm[i])
                else:
                    # Compute the cross-spectrum
                    power_spectra_array[i,j] = self.compute_cross_spectrum(maps[i], self.allfwhm[i], maps[j], self.allfwhm[j])
        return power_spectra_array

    def compute_power_spectra(self):
        '''
        Function to compute the power spectra array for the sky and for the noise realisations

        Return :
            - sky power spectra array (array [nrec/ncomp, nrec/ncomp])
            - noise power spectra array (array [nrec/ncomp, nrec/ncomp])
        '''

        sky_power_spectra = self.compute_array_power_spectra(self.sky_maps)
        noise_power_spectra = self.compute_array_power_spectra(self.noise_maps)
        return sky_power_spectra, noise_power_spectra


class SyntheticBeam(object):
    pass


class NamasterEll:
    '''
    Class to compute the ell list using NamasterLib and to initialize NamasterLib
    '''

    def __init__(self, iter):

        with open('spectrum_config.yml', "r") as stream:
            self.param_spectrum = yaml.safe_load(stream)
        self.path_sky = self.param_spectrum['data']['path_sky']
        self.iter = iter

    def find_data(self, path):
        '''
        Function to extract the pickle file of one realisation. Useful to have access to the realisations' parameters
        '''

        data_names = os.listdir(path)

        one_realisation = pickle.load(open(path + '/' + data_names[self.iter], 'rb'))

        return one_realisation

    def ell(self):
        
        realisation = self.find_data(self.path_sky)

        # Import simulation parameters
        simu_parameters = realisation['parameters']
        nside = simu_parameters['Sky']['nside']

        # Call the Namaster class & create the ell list 
        coverage = realisation['coverage']
        seenpix = coverage/np.max(coverage) < 0.2
        lmin, lmax, delta_ell = self.param_spectrum['Spectrum']['lmin'], 2*nside-1, self.param_spectrum['Spectrum']['dl']
        namaster = nam.Namaster(weight_mask = list(~np.array(seenpix)), lmin = lmin, lmax = lmax, delta_ell = delta_ell)

        ell = namaster.get_binning(nside)[0]
        
        return ell, namaster


def find_data(path, iter):
        '''
        Function to extract the pickle file of one realisation. Useful to have access to the realisations' parameters

        Return :
            - pkl file dictionary (dict)
        '''

        data_names = os.listdir(path)

        one_realisation = pickle.load(open(path + '/' + data_names[iter], 'rb'))

        return one_realisation

def save(iteration):
    '''
    Function to compute the power spectra and to save the results in a pickle file
    '''

    with open('spectrum_config.yml', "r") as stream:
        param = yaml.safe_load(stream)
    config = param['simu']['qubic_config']
    nrec = param['simu']['nrec']
    path = param['data']['path_out']
    path_sky = param['data']['path_sky']
    one_realisation = find_data(path_sky, iteration)
    simu_parameters = Spectra(iteration).find_data(path_sky)['parameters']
    sky_power_spectra, noise_power_spectra = Spectra(iteration).compute_power_spectra()
    print('nrec', nrec)
    pkl_path = path + f'{config}_' + f'Nrec={nrec}_spectra/'

    if not os.path.isdir(pkl_path):
            os.makedirs(pkl_path)

    with open(pkl_path + f'{iteration}.pkl', 'wb') as handle:
        pickle.dump({'sky_ps':sky_power_spectra, 'noise_ps':noise_power_spectra, 'parameters':simu_parameters, 'coverage':one_realisation['coverage']}, handle, protocol=pickle.HIGHEST_PROTOCOL)

iter = int(sys.argv[1])
save(iter)