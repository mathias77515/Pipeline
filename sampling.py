#### General packages
import pickle
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import scipy
import healpy as hp
import emcee
import yaml
from multiprocessing import Pool
from getdist import plots, MCSamples
import getdist
import time

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
import fgb.component_model as c
import fgb.mixing_matrix as mm
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pipeline import *
from pyoperators import *
from preset.preset import *

#### Nested Sampling packages
import dynesty
from dynesty import plotting as dyplot
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc

class DataFMM:
    """
    Class to manipulate data from your simulations
    """

    def __init__(self):

        with open('sampling_config.yml', "r") as stream:
            self.param_sampling = yaml.safe_load(stream)
        self.config = self.param_sampling['simu']['qubic_config']
        self.path = self.param_sampling['data']['path']
        self.path_sky = self.param_sampling['data']['path_sky']
        self.path_noise = self.param_sampling['data']['path_noise']
        self.observation_name = self.param_sampling['simu']['name']
        self.nrec = self.param_sampling['simu']['nrec']
        self.data = self.find_data()
        self.nsub = self.data['parameters']['QUBIC']['nsub']
        self.nside = self.data['parameters']['Sky']['nside']
        self.fsub = int(self.nsub / self.nrec)
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()

    def find_data(self):
        '''
        Function to extract the pickle file of one realisation. Useful to have access to the realisations' parameters
        '''

        data_names = os.listdir(self.path_sky)

        one_realisation = pickle.load(open(self.path_sky + '/' + data_names[0], 'rb'))

        return one_realisation

    def compute_data(self, path):
        '''
        Function to compute the mean and std on your spectra

        Argument : 
            - path(str): give the file's path where your realisations are stored
        '''

        data_names = os.listdir(path)
        # Store power spectra and maps, then return the power spectra's mean and std and the maps associated with the path given
        ps_data = []
        maps_data = []
        for realisation in range(0, self.param_sampling['data']['n_real']):
            data = pickle.load(open(path + '/' + data_names[realisation], 'rb'))
            ps_data.append(data['Dl'])
            maps_data.append(data['maps'])
        ps_data = np.reshape(ps_data, [self.param_sampling['data']['n_real'],self.param_sampling['simu']['nrec'], self.param_sampling['simu']['nrec'], np.shape(data['Dl'][0])[0]])
        
        self.nus = data['nus']

        # Compute Mean & Error on each realisations of PSs 
        mean_ps_data = np.mean(ps_data, axis = 0)
        error_ps_data = np.std(ps_data, axis = 0)

        return (mean_ps_data, error_ps_data, maps_data)

    def get_ultrawideband_config(self):
        """
        Method that pre-compute UWB configuration.
        """

        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave

        return nu_ave, 2*delta/nu_ave
    
    def get_dict(self):
        """
        Method to modify the qubic dictionary.
        """

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()
        params = self.data['parameters']

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

    # def cross_sectra_convo(self, tab_power_spectra, maps_data):
    #     '''
    #     FUnction that will compute the cross-spectra between each reconsructed sub-bands taking into accounts the different convolutions between each maps
    #     '''

    #     my_dict, _ = self.get_dict()
    #     joint = acq.JointAcquisitionFrequencyMapMaking(my_dict, self.data['parameters']['QUBIC']['type'], self.data['parameters']['QUBIC']['nrec'], self.data['parameters']['QUBIC']['nsub'])
    #     allfwhm = joint.qubic.allfwhm
    #     _, namaster = NamasterEll().ell()

    #     for i in range(self.nrec):
    #         for j in range(self.nrec):
    #             if i != j:
    #                 for real in range(self.param_sampling['data']['n_real']):
    #                     cross_spect = []
    #                     # Put the map with the highest resolution at the worst one before doing the cross spectrum
    #                     if allfwhm[i*self.fsub]<allfwhm[j*self.fsub] :
    #                         C = HealpixConvolutionGaussianOperator(fwhm=allfwhm[j*self.fsub])
    #                         convoluted_map = C*maps_data[real][i]
    #                         cross_spect.append(namaster.get_spectra(map=convoluted_map.T, map2=maps_data[real][j].T)[1][:, 2])
    #                     else:
    #                         C = HealpixConvolutionGaussianOperator(fwhm=allfwhm[i*self.fsub])
    #                         convoluted_map = C*maps_data[real][j]
    #                         cross_spect.append(namaster.get_spectra(map=maps_data[real][i].T, map2=convoluted_map.T)[1][:, 2])
    #                 tab_power_spectra[i, j, :] = np.mean(cross_spect, axis=0)
        
    #     return  tab_power_spectra

    def get_array_power_spectra(self):
        '''
        Function to compute the auto-spectra and cross-spectra for each realisation, for the sky and the noise and that will retrun their mean and std, using Spectra class
        '''

        # mean_ps_sky, error_ps_sky, maps_sky = self.compute_data(self.path_sky)

        # if self.data['parameters']['QUBIC']['convolution']:
        #     mean_ps_sky = self.cross_sectra_convo(mean_ps_sky, maps_sky)

        # if self.param_sampling['simu']['noise']:
        #     mean_ps_noise, error_ps_noise, _ = self.compute_data(self.path_noise)
        #     mean_ps_sky = self.auto_spectra_noise_reduction(mean_ps_sky, mean_ps_noise)

        #     return (mean_ps_sky, error_ps_noise, error_ps_sky)
        
        # return mean_ps_sky, np.ones((np.shape(mean_ps_sky))), error_ps_sky

        _, _, maps_sky = self.compute_data(self.path_sky)
        _, _, maps_noise = self.compute_data(self.path_noise)
        mean_ps_sky, error_ps_sky = Spectra().compute_array_power_spectra(maps_sky)
        mean_ps_noise, error_ps_noise = Spectra().compute_array_power_spectra(maps_noise)

        mean_ps_sky = Spectra().auto_spectra_noise_reduction(mean_ps_sky, mean_ps_noise)

        return mean_ps_sky, error_ps_noise, error_ps_sky


class DataCMM:
    """
    Class to manipulate data from your simulations
    """

    def __init__(self):

        with open('sampling_config.yml', "r") as stream:
            self.param_sampling = yaml.safe_load(stream)
        self.config = self.param_sampling['simu']['qubic_config']
        self.path = self.param_sampling['data']['path']
        self.path_sky = self.param_sampling['data']['path_sky']
        self.path_noise = self.param_sampling['data']['path_noise']
        self.observation_name = self.param_sampling['simu']['name']
        self.nrec = self.param_sampling['simu']['nrec']
        self.data = self.find_data()
        self.nsub = self.data['parameters']['MapMaking']['qubic']['nsub']
        self.nside = self.data['parameters']['MapMaking']['qubic']['nside']
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()

    def find_data(self):
        '''
        Function to extract the pickle file of one realisation. Useful to have access to the realisations' parameters
        '''

        data_names = os.listdir(self.path_sky)

        data = pickle.load(open(self.path_sky + '/' + data_names[0], 'rb'))

        return data

    def compute_data(self, path):
        '''
        Function to compute the mean and std on your spectra

        Argument : 
            - path(str): give the file's path where your realisations are stored
        '''

        data_names = os.listdir(path)
        # Store the datas into arrays
        map_data = []
        for realisation in range(0, self.param_sampling['data']['n_real']):
            data = pickle.load(open(path + '/' + data_names[realisation], 'rb'))
            map_data.append(data['components'])

        return map_data

    def get_power_spectra(self):
        '''
        Function to compute the auto-spectra and cross-spectra for each realisation, for the sky and the noise and that will retrun their mean and std, using Spectra class
        '''

        _, _, maps_sky = self.compute_data(self.path_sky)
        _, _, maps_noise = self.compute_data(self.path_noise)
        mean_ps_sky, error_ps_sky = Spectra().compute_array_power_spectra(maps_sky)
        mean_ps_noise, error_ps_noise = Spectra().compute_array_power_spectra(maps_noise)

        mean_ps_sky = Spectra().auto_spectra_noise_reduction(mean_ps_sky, mean_ps_noise)

        return mean_ps_sky, error_ps_noise, error_ps_sky

class Spectra:
    '''
    Class to compute the different spectra for our realisations
    '''

    def __init__(self):

        with open('sampling_config.yml', "r") as stream:
            self.param_sampling = yaml.safe_load(stream)
        self.pipeline = self.param_sampling['data']['pipeline']
        self.n_real = self.param_sampling['data']['n_real']
        if self.pipeline == 'CMM':
            self.data_parameters = DataCMM().find_data()['parameters']
            self.components_maps = DataCMM().compute_data(self.param_sampling['data']['path_sky'])
            self.ncomp = np.shape(self.components_maps[0])[0]
            self.nsub = self.data_parameters['QUBIC']['nsub']
            self.allfwhm = self.sims.joint.qubic.allfwhm
        if self.pipeline == 'FMM':
            self.data = DataFMM().find_data()
            self.data_parameters = self.data['parameters']
            self.nrec = self.data_parameters['QUBIC']['nrec']
            self.nsub = self.data_parameters['QUBIC']['nsub']
            self.fsub = int(self.nsub / self.nrec)
            self.data_parameters = DataFMM().find_data()['parameters']
            self.mean_ps_sky, self.error_ps_sky, self.maps_sky = DataFMM().compute_data(self.param_sampling['data']['path_sky'])
            self.mean_ps_noise, self.error_ps_noise, _ = DataFMM().compute_data(self.param_sampling['data']['path_noise'])
            my_dict, _ = DataFMM().get_dict()
            joint = acq.JointAcquisitionFrequencyMapMaking(my_dict, self.data_parameters['QUBIC']['type'], self.nrec, self.nsub)
            self.allfwhm = joint.qubic.allfwhm
        _, self.namaster = NamasterEll().ell()

    def compute_auto_spectrum(self, map):
        '''
        Function to compute the auto-spectrum of a given map

        Argument : 
            - map(array) : (nrec/ncomp, npix, nstokes)
        '''

        DlBB = self.namaster.get_spectra(map=map.T, map2=None)[1][:, 2]
        
        return DlBB

    def auto_spectra_noise_reduction(self, mean_data, mean_noise):
        '''
        Function to remove the mean of the noise realisations to the auto-spectra

        Arguments :
            - mean_data(array) : (nrec/ncomp, nrec/ncomp, ell) array that will contain the mean of all the auto and cross spectra of the sky realisations
            - mean_noise(array) : (nrec/ncomp, nrec/ncomp, ell) array that will contain the mean of all the auto and cross spectra of the noise realisation
        '''

        for i in range(np.shape(mean_data)[0]):
            mean_data[i, i, :] -= mean_noise[i, i, :]

        return mean_data

    def compute_cross_spectrum(self, map1, fwhm1, map2, fwhm2):
        '''
        Function to compute cross-spectrum
        '''

        # Put the map with the highest resolution at the worst one before doing the cross spectrum
        if fwhm1<fwhm2 :
            C = HealpixConvolutionGaussianOperator(fwhm=fwhm2)
            convoluted_map = C*map1
            return self.namaster.get_spectra(map=convoluted_map.T, map2=map2.T)[1][:, 2]
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=fwhm1)
            convoluted_map = C*map2
            return self.namaster.get_spectra(map=map1.T, map2=convoluted_map.T)[1][:, 2]

    def compute_array_power_spectra(self, maps):
        '''
        Function to build an array fill with all the power spectra computed
        '''

        if self.pipeline == 'FMM':
            idx_lenght = self.nrec
        elif self.pipeline == 'CMM':
            idx_lenght = self.ncomp
        power_spectra_array = np.zeros((self.n_real, idx_lenght,idx_lenght))
            for i in range(idx_lenght):
                for j in range(0, i + 1):
                    if i==j :
                        # Compute the auto-spectrum
                        for realisation in range(self.n_real):
                            power_spectra_array[realisation,i,j] = self.compute_auto_spectrum(maps[realisation][i])
                    else:
                        # Compute the cross-spectrum
                        for realisation in range(self.n_real):
                            power_spectra_array[realisation,i,j] = self.compute_cross_spectrum(maps[realisation][i], self.allfwhm[i*self.fsub],maps[realisation][j], self.allfwhm[j*self.fsub])

            return np.mean(power_spectra_array, axis=0), np.std(power_spectra_array, axis=0)

class NamasterEll:
    '''
    Class to compute the ell list using NamasterLib
    '''

    def __init__(self):

        with open('sampling_config.yml', "r") as stream:
            self.param_sampling = yaml.safe_load(stream)

    def ell(self):
        
        realisation = DataFMM().find_data()

        # Import simulation parameters
        simu_parameters = realisation['parameters']
        nside = simu_parameters['Sky']['nside']

        # Call the Namaster class & create the ell list 
        coverage = realisation['coverage']
        seenpix = coverage/np.max(coverage) < 0.2
        lmin, lmax, delta_ell = simu_parameters['Spectrum']['lmin'], 2*nside-1, simu_parameters['Spectrum']['dl']
        namaster = nam.Namaster(weight_mask = list(~np.array(seenpix)), lmin = lmin, lmax = lmax, delta_ell = delta_ell)

        ell = namaster.get_binning(nside)[0]
        
        return ell, namaster


class CMB:
    '''
    Class to define the CMB model
    '''

    def __init__(self, ell):
        
        self.ell = ell

    def cl_to_dl(self, cl):
        '''
        Function to convert the cls into the dls
        '''

        dl = np.zeros(self.ell.shape[0])
        for i in range(self.ell.shape[0]):
            dl[i] = (self.ell[i]*(self.ell[i]+1)*cl[i])/(2*np.pi)
        return dl

    def get_pw_from_planck(self, r, Alens):
        '''
        Function to compute the CMB power spectrum from the Planck data
        '''

        CMB_CL_FILE = op.join('/sps/qubic/Users/TomLaclavere/mypackages/Cls_Planck2018_%s.fits')
        power_spectrum = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000]
        
        if Alens != 1.:
            power_spectrum[2] *= Alens
        
        if r:
            power_spectrum += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[:,:4000]
        
        return np.interp(self.ell, np.linspace(1, 4001, 4000), power_spectrum[2])

    def model_cmb(self, r, Alens):
        '''
        Define the CMB model, depending on r and Alens
        '''

        dlBB = self.cl_to_dl(self.get_pw_from_planck(r, Alens))
        return dlBB


class Dust:
    '''
    Function to define the Dust model
    '''

    def __init__(self, ell):
        
        self.ell = ell
        realisation = DataFMM().find_data()
        self.nus = realisation['nus']
        self.nrec = len(self.nus)

    def scale_dust(self, nu, nu0_d, betad, temp=20):
        '''
        Function to compute the dust mixing matrix element, depending on the frequency
        '''

        comp = c.Dust(nu0 = nu0_d, temp=temp, beta_d = betad)
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]

        return A

    def fnus_dust(self, nus, nu0_d, betad):
        '''
        Function to compute the mixing matrix elements for all the frequencies considered in your realisations
        '''

        fnus = np.zeros(self.nrec)
        for nu_index in range(self.nrec):
            fnus[nu_index] = self.scale_dust(nus[nu_index], nu0_d, betad)

        return fnus

    def model_dust_frequency(self, Ad, alphad, deltad, fnu1, fnu2):
        '''
        Function to define the Dust model for two frequencies
        '''

        return Ad * deltad * fnu1 * fnu2 * (self.ell/80)**alphad

    def model_dust(self, Ad, alphad, betad, deltad, nu0_d):
        '''
        Function defining the Dust model for all frequencies, depending on Ad, alphad, betad, deltad & nu0_d
        '''
        
        fnus = self.fnus_dust(self.nus, nu0_d, betad)

        models = np.zeros((self.nrec, self.nrec, len(self.ell)))
        for i in range(self.nrec):
            for j in range(self.nrec):
                models[i][j][:] = self.model_dust_frequency(Ad, alphad, deltad, fnus[i], fnus[j])
        return models


class MCMC:
    '''
    Class to perform MCMC on the chosen sky parameters
    '''

    def __init__(self):

        with open('sampling_config.yml', "r") as stream:
            self.param_sampling = yaml.safe_load(stream)
        self.ell, _ = NamasterEll().ell()
        realisation = DataFMM().find_data()
        self.nus = realisation['nus']
        self.sky_parameters = self.param_sampling['SKY_PARAMETERS']
        self.ndim, self.sky_parameters_fitted_names, self.sky_parameters_all_names = self.ndim_and_parameters_names()
        self.mean_ps_sky, self.error_ps_noise, self.error_ps_sky = DataFMM().get_array_power_spectra()
        print('SHAAAAAAPE', np.shape(self.mean_ps_sky))

    def ndim_and_parameters_names(self):
        '''
        Function to create the name list of the parameter(s) that you want to find with the MCMC and to compute the number of these parameters
        '''
        
        ndim = 0   
        sky_parameters_fitted_names = []   
        sky_parameters_all_names = []

        for parameter in self.sky_parameters:
            sky_parameters_all_names.append(parameter)
            if self.sky_parameters[parameter][0] is True:
                ndim += 1
                sky_parameters_fitted_names.append(parameter)

        return ndim, sky_parameters_fitted_names, sky_parameters_all_names

    def initial_conditions(self):
        '''
        Function to computes the MCMC initial conditions
        '''

        nwalkers = self.param_sampling['MCMC']['nwalkers']

        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            for j in range(self.ndim):
                name = self.sky_parameters_fitted_names[j]
                p0[i,j] = np.random.random() * self.param_sampling['SKY_PARAMETERS'][name][2] - self.param_sampling['SKY_PARAMETERS'][name][1]

        return p0

    def prior(self, x):
        '''
        Function to define priors to help the MCMC convergence
        '''
        
        for isky_param, sky_param in enumerate(x):
            name = self.sky_parameters_fitted_names[isky_param]

            if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                return - np.inf

        return 0

    def chi2(self, tab):
        '''
        chi2 function
        '''

        tab_parameters = np.zeros(len(self.param_sampling['SKY_PARAMETERS']))
        cpt = 0

        for i, iname in enumerate(self.param_sampling['SKY_PARAMETERS']):
            if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
                tab_parameters[i] = self.param_sampling['SKY_PARAMETERS'][iname][0]
            else:
                tab_parameters[i] = tab[cpt]
                cpt += 1

        # for i, iname in enumerate(self.param_sampling['SKY_PARAMETERS']):
        #     if iname == 'r':
        #         if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
        #             r = self.param_sampling['SKY_PARAMETERS'][iname][0]
        #     elif iname == 'Alens':
        #         if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
        #             Alens = self.param_sampling['SKY_PARAMETERS'][iname][0]
        #     elif iname == 'Ad':
        #         if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
        #             Ad = self.param_sampling['SKY_PARAMETERS'][iname][0]
        #     elif iname == 'betad':
        #         if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
        #             betad = self.param_sampling['SKY_PARAMETERS'][iname][0]
        #     elif iname == 'alphad':
        #         if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
        #             alphad = self.param_sampling['SKY_PARAMETERS'][iname][0]
        #     elif iname == 'deltad':
        #         if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
        #             deltad = self.param_sampling['SKY_PARAMETERS'][iname][0]
        #     elif iname == 'nu0_d':
        #         if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
        #             nu0_d = self.param_sampling['SKY_PARAMETERS'][iname][0]

        # # Add the parameters you want to find
        # sky_parameters_fitted_names = self.sky_parameters_fitted_names
        # for isky_param, sky_param in enumerate(tab):
        #     if sky_parameters_fitted_names[isky_param] == 'r':
        #         r = sky_param
        #     elif sky_parameters_fitted_names[isky_param] == 'Alens':
        #         Alens = sky_param
        #     elif sky_parameters_fitted_names[isky_param] == 'Ad':
        #         Ad = sky_param
        #     elif sky_parameters_fitted_names[isky_param] == 'betad':
        #         betad = sky_param
        #     elif sky_parameters_fitted_names[isky_param] == 'alphad':
        #         alphad = sky_param
        #     elif sky_parameters_fitted_names[isky_param] == 'deltad':
        #         deltad = sky_param
        #     elif sky_parameters_fitted_names[isky_param] == 'nu0_d':
        #         nu0_d = sky_param
                
        # Return the chi2 function

        r, Alens, nu0_d, Ad, alphad, betad, deltad = tab_parameters

        if self.param_sampling['simu']['name'] == 'CMB':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_ps_sky - CMB(self.ell).model_cmb(r, Alens))/(self.error_ps_noise))**2)
        if self.param_sampling['simu']['name'] == 'Dust':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_ps_sky - Dust(self.ell).model_dust(Ad, alphad, betad, deltad, nu0_d))/(self.error_ps_noise))**2)
        if self.param_sampling['simu']['name'] == 'Sky':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_ps_sky - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell).model_dust(Ad, alphad, betad, deltad, nu0_d)))/(self.error_ps_noise))**2)

    def __call__(self):
        '''
        Funtion to perform the MCMC and save the results
        '''

        # Define the MCMC parameters, initial conditions and ell list
        nwalkers = self.param_sampling['MCMC']['nwalkers']
        mcmc_steps = self.param_sampling['MCMC']['mcmc_steps']
        p0 = self.initial_conditions()
        ell = self.ell
        print(DataFMM().find_data()['parameters'])
        
        # Start the MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_prob_fn = self.chi2, pool = pool, moves = [(emcee.moves.StretchMove(), self.param_sampling['MCMC']['stretch_move_factor']), (emcee.moves.DESnookerMove(gammas=self.param_sampling['MCMC']['snooker_move_gamma']), 1 - self.param_sampling['MCMC']['stretch_move_factor'])])
            sampler.run_mcmc(p0, mcmc_steps, progress=True)

        samples_flat = sampler.get_chain(flat = True, discard = self.param_sampling['MCMC']['discard'], thin = self.param_sampling['MCMC']['thin'])
        samples = sampler.get_chain()

        # Plot the walkers
        fig, ax = plt.subplots(1, self.ndim, figsize = (15, 5))
        for j in range(self.ndim):
            for i in range(nwalkers):
                ax[j].plot(samples[:, i, j])
                ax[j].set_title(self.sky_parameters_fitted_names[j])

        config = self.param_sampling['simu']['qubic_config']
        nrec = self.param_sampling['simu']['nrec']
        n_real = self.param_sampling['data']['n_real']
        path_plot = f'{config}_Nrec={nrec}_plots_MCMC'
        if not os.path.isdir(path_plot):
            os.makedirs(path_plot)
        plt.savefig(self.param_sampling['data']['path'] + path_plot + f'/walkers_plot_Nreal={n_real}')

        # Triangle plot
        plt.figure()
        s = MCSamples(samples=samples_flat, names=self.sky_parameters_fitted_names, labels=self.sky_parameters_fitted_names)
        g = plots.get_subplot_plotter(width_inch=10)
        g.triangle_plot([s], filled=True, title_limit=1)
        for ax in g.subplots[:,0]:
            ax.axvline(0, color='gray')

        path_plot_triangle = f'{config}_Nrec={nrec}_plots'
        plt.savefig(self.param_sampling['data']['path'] + path_plot + f'/triangle_plot_Nreal={n_real}')

        # Data vs Fit plot
        plt.figure()
        mcmc_values = np.mean(samples_flat, axis=0)
        parameters_values = []
        cpt=0
        for parameter in self.sky_parameters:
            if self.sky_parameters[parameter][0] is True:
                parameters_values.append(mcmc_values[cpt])
                cpt+=1
            else:
                parameters_values.append(self.sky_parameters[parameter][0])
        Dl_mcmc = CMB(self.ell).model_cmb(parameters_values[0], parameters_values[1]) + Dust(self.ell).model_dust(parameters_values[3], parameters_values[4], parameters_values[5], parameters_values[6], parameters_values[2])
        plt.plot(self.ell[:5], Dl_mcmc[0][0][:5], label = 'MCMC')
        plt.errorbar(self.ell[:5], self.mean_ps_sky[0][0][:5], self.error_ps_sky[0][0][:5], label = 'Data')
        plt.legend()
        plt.xlabel('l')
        plt.ylabel('Dl')
        plt.title('CMB + Dust spectrum')          
        plt.savefig(self.param_sampling['data']['path'] + path_plot + f'/Comparison_plot_Nreal={n_real}')

class NestedSampling:
    '''
    Class to perform Nested Sampling in our sky parameters
    '''

    def __init__(self):

        with open('sampling_config.yml', "r") as stream:
            self.param_sampling = yaml.safe_load(stream)
        self.ell, _ = NamasterEll().ell()
        data = DataFMM().find_data()
        self.nus = data['nus']
        self.sky_parameters = self.param_sampling['SKY_PARAMETERS']
        self.ndim, self.sky_parameters_fitted_names, self.sky_parameters_all_names = self.ndim_and_parameters_names()
        self.mean_ps_sky, self.error_ps_noise, self.error_ps_sky = DataFMM().get_array_power_spectra()

    def ndim_and_parameters_names(self):
        '''
        Function to create the name list of the parameter(s) that you want to find with the MCMC and to compute the number these parameters
        '''
        
        ndim = 0   
        sky_parameters_fitted_names = []   
        sky_parameters_all_names = []

        for parameter in self.sky_parameters:
            sky_parameters_all_names.append(parameter)
            if self.sky_parameters[parameter][0] is True:
                ndim += 1
                sky_parameters_fitted_names.append(parameter)

        return ndim, sky_parameters_fitted_names, sky_parameters_all_names

    def chi2(self, tab):
        '''
        chi2 function
        '''

        tab_parameters = np.zeros(len(self.param_sampling['SKY_PARAMETERS']))
        cpt = 0

        for i, iname in enumerate(self.param_sampling['SKY_PARAMETERS']):
            if self.param_sampling['SKY_PARAMETERS'][iname][0] is not True:
                tab_parameters[i] = self.param_sampling['SKY_PARAMETERS'][iname][0]
            else:
                tab_parameters[i] = tab[cpt]
                cpt += 1
                
        # Return the chi2 function

        r, Alens, nu0_d, Ad, alphad, betad, deltad = tab_parameters

        if self.param_sampling['simu']['name'] == 'CMB':
            return - 0.5 * np.sum(((self.mean_ps_sky - CMB(self.ell).model_cmb(r, Alens))/(self.error_ps_noise))**2)
        if self.param_sampling['simu']['name'] == 'Dust':
            return - 0.5 * np.sum(((self.mean_ps_sky - Dust(self.ell).model_dust(Ad, alphad, betad, deltad, nu0_d))/(self.error_ps_noise))**2)
        if self.param_sampling['simu']['name'] == 'Sky':
            return - 0.5 * np.sum(((self.mean_ps_sky - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell).model_dust(Ad, alphad, betad, deltad, nu0_d)))/(self.error_ps_noise))**2)

    def ptform_uniform(self, u):
        '''
        Function to perform an uniform prior transform for the Nested Sampling
        '''

        ptform = []
        cpt = 0
        for iname in self.sky_parameters_all_names:
            if self.param_sampling['SKY_PARAMETERS'][iname][0] is True:
                ptform.append(u[cpt]*self.param_sampling['SKY_PARAMETERS'][iname][2] - self.param_sampling['SKY_PARAMETERS'][iname][1])
                cpt += 1
        return ptform

    def __call__(self):
        '''
        Funtion to perform the Nested Sampling and save the results
        '''

        nlive = self.param_sampling['NS']['nlive']
        ell = self.ell
        print(DataFMM().find_data()['parameters'])
        
        if self.param_sampling['NS']['DynamicNS'] is True:
            print('Dynamic Nested Sampling !!!')
            with Pool() as pool:
                sampler_ns = DynamicNestedSampler(self.chi2, self.ptform_uniform, self.ndim, pool = pool, nlive = nlive, queue_size=self.param_sampling['NS']['queue_size'], bound=self.param_sampling['NS']['bound'])
                sampler_ns.run_nested()
        else:
            with Pool() as pool:
                print('Nested Sampling !')
                sampler_ns = NestedSampler(self.chi2, self.ptform_uniform, self.ndim, pool = pool, nlive = nlive, queue_size=self.param_sampling['NS']['queue_size'], bound=self.param_sampling['NS']['bound'])
                sampler_ns.run_nested()

        results = sampler_ns.results

        # Plot the traceplots
        fig, axes = dyplot.traceplot(results, show_titles=True, labels = self.sky_parameters_fitted_names,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))

        config = self.param_sampling['simu']['qubic_config']
        nrec = self.param_sampling['simu']['nrec']
        n_real = self.param_sampling['data']['n_real']
        if self.param_sampling['NS']['DynamicNS'] is True:
            path_plot = f'{config}_Nrec={nrec}_plots_DynamicNS'
        else:
            path_plot = f'{config}_Nrec={nrec}_plots_NS'
        if not os.path.isdir(path_plot):
            os.makedirs(path_plot)
        plt.savefig(self.param_sampling['data']['path'] + path_plot + f'/traceplot_Nreal={n_real}')

        # Runplots
        fig, axes = dyplot.runplot(results)
        plt.savefig(self.param_sampling['data']['path'] + path_plot + f'/runplot_Nreal={n_real}')


        # Triangle plot
        fig, axes = plt.subplots(3, 3)
        axes = axes.reshape((3, 3))
        fg, ax = dyplot.cornerplot(results, color='blue', title_fmt = '.4f', show_titles=True, labels = self.sky_parameters_fitted_names,
                           max_n_ticks=3, quantiles=None,
                           fig=(fig, axes[:, :3]))
        plt.savefig(self.param_sampling['data']['path'] + path_plot + f'/triangle_plot_Nreal={n_real}')

        # Data vs Fit plot
        plt.figure()
        samples, weights = results.samples, results.importance_weights()
        mean_ns, cov_ns = dyfunc.mean_and_cov(samples, weights)
        parameters_values = []
        cpt=0
        for parameter in self.sky_parameters:
            if self.sky_parameters[parameter][0] is True:
                parameters_values.append(mean_ns[cpt])
                cpt+=1
            else:
                parameters_values.append(self.sky_parameters[parameter][0])
        Dl_ns = CMB(self.ell).model_cmb(parameters_values[0], parameters_values[1]) + Dust(self.ell).model_dust(parameters_values[3], parameters_values[4], parameters_values[5], parameters_values[6], parameters_values[2])
        plt.plot(self.ell[:5], Dl_ns[0][0][:5], label = 'NestedSampling')
        plt.errorbar(self.ell[:5], self.mean_ps_sky[0][0][:5], self.error_ps_sky[0][0][:5], label = 'Data')
        plt.legend()
        plt.xlabel('l')
        plt.ylabel('Dl')
        plt.title('CMB + Dust spectrum')          
        plt.savefig(self.param_sampling['data']['path'] + path_plot + f'/Comparison_plot_Nreal={n_real}')

with open('sampling_config.yml', "r") as stream:
    param = yaml.safe_load(stream)

if param['Method'] == 'MCMC':
    print("Chosen method = MCMC")
    MCMC()()
elif param['Method'] == 'NS':
    print("Chosen method = Nested Sampling")
    NestedSampling()()
else:
    print('Wrong sampling method')

print("Fitting done")













