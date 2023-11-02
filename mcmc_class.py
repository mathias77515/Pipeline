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
from qubic import mcmc
from qubic import AnalysisMC as amc
import fgb.component_model as c
import fgb.mixing_matrix as mm


class Data:
    """
    Class to manipulate data from your simulations
    """

    def __init__(self):

        with open('mcmc_config.yml', "r") as stream:
            self.param = yaml.safe_load(stream)
        self.config = self.param['simu']['qubic_config']
        self.path = self.param['data']['path']
        self.name = self.param['simu']['name']
        self.nrec = self.param['simu']['nrec']

    def find_data(self):
        '''
        Function to extract the pickle file of one realisation. Useful to have access to the realisations' parameters
        '''

        path_data = spath_data = self.path + self.config + f'_Nrec={self.nrec}_{self.name}'
        data_names = os.listdir(path_data)

        data = pickle.load(open(path_data + '/' + data_names[0], 'rb'))

        return data

    def compute_data(self, name):
        '''
        Function to compute the mean and std on your spectra

        Argument : 
            - name(str): CMB, Dust or Sky
        '''

        path_data = self.path + self.config + f'_Nrec={self.nrec}_{name}'
        data_names = os.listdir(path_data)

        # Store the datas into arrays
        ps_data = []
        for realisation in range(self.param['data']['n_real']):
            data = pickle.load(open(path_data + '/' + data_names[realisation], 'rb'))
            ps_data.append(data['Dl'])
        ps_data = np.reshape(ps_data, [self.param['data']['n_real'],self.param['simu']['nrec'], self.param['simu']['nrec'], np.shape(data['Dl'][0])[0]])
        
        # Compute Mean & Error on each realisations of Noise & Sky's PSs 
        mean_data = np.mean(ps_data, axis = 0)
        error_data = np.std(ps_data, axis = 0)

        return (mean_data, error_data)

    def auto_spectra_noise_reduction(self, mean_data, mean_noise):
        '''
        Function to remove the mean of the noise realisations to the auto-spectra
        '''

        for i in range(self.param['simu']['nrec']):
            mean_data[i, i, :] -= mean_noise[i, i, :]

        return mean_data

    def data(self):
        '''
        Function to compute the mean of your spectra and the std of the noise realisations, taking into account the noise reduction
        '''

        mean_data, error_data = self.compute_data(self.name)

        if self.param['simu']['noise']:
            mean_noise, error_noise = self.compute_data('Noise')
            mean_data = self.auto_spectra_noise_reduction(mean_data, mean_noise)

            return (mean_data, error_noise)
        
        return mean_data, np.ones((np.shape(mean_data)))


class namaster_ell:
    '''
    Class to compute the l using NamasterLib
    '''

    def __init__(self):

        with open('mcmc_config.yml', "r") as stream:
            self.param = yaml.safe_load(stream)

    def ell(self):

        data = Data().find_data()

        # Import simulation parameters
        simu_parameters = data['parameters']
        nside = simu_parameters['Sky']['nside']

        # Call the Namaster class & create the ell list 
        coverage = data['coverage']
        seenpix = coverage/np.max(coverage) < 0.2
        lmin, lmax, delta_ell = simu_parameters['Spectrum']['lmin'], 2*nside-1, simu_parameters['Spectrum']['dl']
        namaster = nam.Namaster(weight_mask = list(~np.array(seenpix)), lmin = lmin, lmax = lmax, delta_ell = delta_ell)

        ell = namaster.get_binning(nside)[0]
        
        return ell


class CMB:
    '''
    Class defining the CMB model
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
        Function defining the CMB model, depending on r and Alens
        '''

        dlBB = self.cl_to_dl(self.get_pw_from_planck(r, Alens))
        return dlBB


class Dust:
    '''
    Function defining the Dust model
    '''

    def __init__(self, ell):
        
        self.ell = ell
        data = Data().find_data()
        self.nus = data['nus']
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


class mcmc:
    '''
    Class to perform MCMC on the chosen sky parameters
    '''

    def __init__(self):

        with open('mcmc_config.yml', "r") as stream:
            self.param = yaml.safe_load(stream)
        self.ell = namaster_ell().ell()
        data = Data().find_data()
        self.nus = data['nus']
        self.sky_parameters = self.param['SKY_PARAMETERS']
        self.ndim, self.sky_parameters_names = self.ndim_and_parameters_names()
        self.mean_data, self.error_noise = Data().data()



    def ndim_and_parameters_names(self):
        '''
        Function to create the name list of the parameter(s) that you want to find with the MCMC and to compute the number of dimensions for it
        '''
        
        ndim = 0   
        sky_parameters_names = []     

        for parameter in self.sky_parameters:
            if self.sky_parameters[parameter][0] is True:
                ndim += 1
                sky_parameters_names.append(parameter)

        return ndim, sky_parameters_names 

    def initial_conditions(self):
        '''
        Function to computes the initial condition of the MCMC
        '''

        ndim, sky_parameters_names = self.ndim, self.sky_parameters_names
        nwalkers = self.param['MCMC']['nwalkers']

        p0 = np.zeros((nwalkers, ndim))
        for i in range(nwalkers):
            for j in range(ndim):
                name = sky_parameters_names[j]
                if name == 'r':
                    multiplicative_factor = self.param['SKY_PARAMETERS'][name][2]
                    initial_value = self.param['SKY_PARAMETERS'][name][1]
                elif name == 'Alens':
                    multiplicative_factor = self.param['SKY_PARAMETERS'][name][2]
                    initial_value = self.param['SKY_PARAMETERS'][name][1]
                elif name == 'Ad':
                    multiplicative_factor = self.param['SKY_PARAMETERS'][name][2]
                    initial_value = self.param['SKY_PARAMETERS'][name][1]
                elif name == 'betad':
                    multiplicative_factor = self.param['SKY_PARAMETERS'][name][2]
                    initial_value = self.param['SKY_PARAMETERS'][name][1]
                elif name == 'alphad':
                    multiplicative_factor = self.param['SKY_PARAMETERS'][name][2]
                    initial_value = self.param['SKY_PARAMETERS'][name][1]
                elif name == 'deltad':
                    multiplicative_factor = self.param['SKY_PARAMETERS'][name][2]
                    initial_value = self.param['SKY_PARAMETERS'][name][1]
                elif name == 'nu0_d':
                    multiplicative_factor = self.param['SKY_PARAMETERS'][name][2]
                    initial_value = self.param['SKY_PARAMETERS'][name][1]


                p0[i,j] = np.random.randn() * multiplicative_factor + initial_value

        return p0

    def prior(self, x):
        '''
        Function to define priors to help the MCMC convergence
        '''
        
        sky_parameters_names = self.sky_parameters_names

        for isky_param, sky_param in enumerate(x):
            name = sky_parameters_names[isky_param]
            if name == 'r':
                if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                    return - np.inf
            elif name == 'Alens':
                if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                    return - np.inf
            elif name == 'Ad':
                if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                    return - np.inf
            elif name == 'alphad':
                if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                    return - np.inf
            elif name == 'betad':
                if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                    return - np.inf
            elif name == 'deltad':
                if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                    return - np.inf
            elif name == 'nu0_d':
                if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                    return - np.inf
        return 0

    def chi2(self, tab):
        '''
        chi2 function
        '''

        a = time.time()
        for iname in self.param['SKY_PARAMETERS']:
            if iname == 'r':
                if self.param['SKY_PARAMETERS'][iname][0] is not True:
                    r = self.param['SKY_PARAMETERS'][iname][0]
            elif iname == 'Alens':
                if self.param['SKY_PARAMETERS'][iname][0] is not True:
                    Alens = self.param['SKY_PARAMETERS'][iname][0]
            elif iname == 'Ad':
                if self.param['SKY_PARAMETERS'][iname][0] is not True:
                    Ad = self.param['SKY_PARAMETERS'][iname][0]
            elif iname == 'betad':
                if self.param['SKY_PARAMETERS'][iname][0] is not True:
                    betad = self.param['SKY_PARAMETERS'][iname][0]
            elif iname == 'alphad':
                if self.param['SKY_PARAMETERS'][iname][0] is not True:
                    alphad = self.param['SKY_PARAMETERS'][iname][0]
            elif iname == 'deltad':
                if self.param['SKY_PARAMETERS'][iname][0] is not True:
                    deltad = self.param['SKY_PARAMETERS'][iname][0]
            elif iname == 'nu0_d':
                if self.param['SKY_PARAMETERS'][iname][0] is not True:
                    nu0_d = self.param['SKY_PARAMETERS'][iname][0]

        # Add the parameters you want to find
        sky_parameters_names = self.sky_parameters_names
        for isky_param, sky_param in enumerate(tab):
            if sky_parameters_names[isky_param] == 'r':
                r = sky_param
            elif sky_parameters_names[isky_param] == 'Alens':
                Alens = sky_param
            elif sky_parameters_names[isky_param] == 'Ad':
                Ad = sky_param
            elif sky_parameters_names[isky_param] == 'betad':
                betad = sky_param
            elif sky_parameters_names[isky_param] == 'alphad':
                alphad = sky_param
            elif sky_parameters_names[isky_param] == 'deltad':
                deltad = sky_param
            elif sky_parameters_names[isky_param] == 'nu0_d':
                nu0_d = sky_param
                
        # Return the chi2 function
        if self.param['simu']['name'] == 'CMB':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_data - CMB(self.ell).model_cmb(r, Alens))/(self.error_noise))**2)
        if self.param['simu']['name'] == 'Dust':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_data - Dust(self.ell).model_dust(Ad, alphad, betad, deltad, nu0_d))/(self.error_noise))**2)
        if self.param['simu']['name'] == 'Sky':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_data - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell).model_dust(Ad, alphad, betad, deltad, nu0_d)))/(self.error_noise))**2)

    def mcmc(self):
        '''
        Funtion to perform the MCMC and save the results
        '''

        ndim, sky_parameters_names = self.ndim_and_parameters_names()
        nwalkers = self.param['MCMC']['nwalkers']
        mcmc_steps = self.param['MCMC']['mcmc_steps']
        p0 = self.initial_conditions()
        ell = self.ell
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn = self.chi2, pool = pool, moves = [(emcee.moves.StretchMove(), self.param['MCMC']['stretch_move_factor']), (emcee.moves.DESnookerMove(gammas=self.param['MCMC']['snooker_move_gamma']), 1 - self.param['MCMC']['stretch_move_factor'])])
            sampler.run_mcmc(p0, mcmc_steps, progress=True)

        samples_flat = sampler.get_chain(flat = True, discard = self.param['MCMC']['discard'])
        samples = sampler.get_chain()

        # Plot the walkers
        fig, ax = plt.subplots(1, ndim, figsize = (15, 5))
        for j in range(ndim):
            for i in range(nwalkers):
                ax[j].plot(samples[:, i, j])
                ax[j].set_title(sky_parameters_names[j])

        config = self.param['simu']['qubic_config']
        nrec = self.param['simu']['nrec']
        n_real = self.param['data']['n_real']
        path_plot = f'{config}_Nrec={nrec}_plots'
        if not os.path.isdir(path_plot):
            os.makedirs(path_plot)
        plt.savefig(self.param['data']['path'] + path_plot + f'/walkers_plot_Nreal={n_real}')

        # Triangle plot
        plt.figure()
        s = MCSamples(samples=samples_flat, names=sky_parameters_names, labels=sky_parameters_names)
        g = plots.get_subplot_plotter(width_inch=10)
        g.triangle_plot([s], filled=True, title_limit=1)

        path_plot_triangle = f'{config}_Nrec={nrec}_plots'
        plt.savefig(self.param['data']['path'] + path_plot + f'/triangle_plot_Nreal={n_real}')

mcmc().mcmc()















