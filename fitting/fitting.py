#### General packages
import pickle
import os
import os.path as op
import sys
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
sys.path.append('/pbs/home/t/tlaclave/sps/Pipeline')

#### QUBIC packages
import qubic
from qubic import NamasterLib as nam
from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray
from qubic import fibtools as ft
from qubic import SpectroImLib as si
import mapmaking.systematics as acq
from qubic import mcmc
from qubic import AnalysisMC as amc
import fgb.component_model as c
import fgb.mixing_matrix as mm
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pipeline import *
from pyoperators import *
#from preset.preset import *

#### Nested fitting packages
import dynesty
from dynesty import plotting as dyplot
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc

class data:
    '''
    Class to extract of the power spectra computed with spectrum.py and to compute useful things
    '''

    def __init__(self):

        with open('fitting_config.yml', "r") as stream:
            self.param = yaml.safe_load(stream)
        self.path_spectra = self.param['data']['path']
        self.power_spectra_sky, self.power_spectra_noise, self.simu_parameters, self.coverage = self.import_power_spectra(self.path_spectra)
        self.nsub = self.simu_parameters['QUBIC']['nsub']
        self.nrec = self.simu_parameters['QUBIC']['nrec']
        self.nreal = self.param['data']['n_real']
        _, allnus150, _, _, _, _ = qubic.compute_freq(150, Nfreq=int(self.nsub/2)-1, relative_bandwidth=0.25)
        _, allnus220, _, _, _, _ = qubic.compute_freq(220, Nfreq=int(self.nsub/2)-1, relative_bandwidth=0.25)
        self.allnus = np.array(list(allnus150) + list(allnus220))
        self.nus = self.average_nus()
        self.mean_ps_sky, self.error_ps_sky = self.compute_mean_std(self.power_spectra_sky)
        self.mean_ps_noise, self.error_ps_noise = self.compute_mean_std(self.power_spectra_noise)
        if self.param['simu']['noise'] is True:
            self.mean_ps_sky = self.spectra_noise_correction(self.mean_ps_sky, self.mean_ps_noise)
        
    def import_power_spectra(self, path):
        '''
        Function to import all the power spectra computed with spectrum.py and store in pickle files

        Argument :
            - path (str) : path to indicate where the pkl files are

        Return :
            - sky power spectra (array) [nreal, nrec/ncomp, nrec/ncomp, len(ell)]
            - noise power spectra (array) [nreal, nrec/ncomp, nrec/ncomp, len(ell)]
            - simulations parameters (dict)
            - simulations coverage (array)
            - bands frequencies for FMM (array) [nrec]
        '''

        power_spectra_sky, power_spectra_noise = [], []
        names = os.listdir(path)
        for i in range(self.param['data']['n_real']):
            ps = pickle.load(open(path + '/' + names[i], 'rb'))
            power_spectra_sky.append(ps['sky_ps'])
            power_spectra_noise.append(ps['noise_ps'])
        return power_spectra_sky, power_spectra_noise, ps['parameters'], ps['coverage']
    
    def average_nus(self):
        
        nus_eff = []
        f = int(self.nsub / self.nrec)
        for i in range(self.nrec):
            nus_eff += [np.mean(self.allnus[i*f : (i+1)*f], axis=0)]
        return np.array(nus_eff)

    def compute_mean_std(self, ps):
        '''
        Function to compute the mean ans the std on our power spectra realisations

        Argument : 
            - power spectra array (array) [nreal, nrec/ncomp, nrec/ncomp, len(ell)]

        Return :
            - mean (array) [nrec/ncomp, nrec/ncomp, len(ell)]
            - std (array) [nrec/ncomp, nrec/ncomp, len(ell)]
        '''

        return np.mean(ps, axis = 0), np.std(ps, axis = 0)

    def spectra_noise_correction(self, mean_data, mean_noise):
        '''
        Function to remove the mean of the noise realisations to the spectra computed

        Arguments :
            - mean sky power spectra (array) [nrec/ncomp, nrec/ncomp, len(ell)] : array that will contain the mean of all the auto and cross spectra of the sky realisations
            - mean noise power spectra (array) [nrec/ncomp, nrec/ncomp, len(ell)] : array that will contain the mean of all the auto and cross spectra of the noise realisation
        
        Return :
            - corrected mean sky power spectra (array) [nrec/ncomp, nrec/ncomp, len(ell)]
        '''

        for i in range(np.shape(mean_data)[0]):
            for j in range(np.shape(mean_data)[1]):
                #if i==j:
                mean_data[i, j, :] -= mean_noise[i, j, :]
        return mean_data


class NamasterEll(data):
    '''
    Class to compute the ell list using NamasterLib
    '''

    def __init__(self):

        with open('fitting_config.yml', "r") as stream:
            self.param_fitting = yaml.safe_load(stream)
        data.__init__(self)

    def ell(self):
        
        nside = self.simu_parameters['Sky']['nside']

        # Call the Namaster class & create the ell list 
        seenpix = self.coverage/np.max(self.coverage) < 0.2
        lmin, lmax, delta_ell = self.simu_parameters['Spectrum']['lmin'], 2*nside-1, self.simu_parameters['Spectrum']['dl']
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

    def __init__(self, ell, nus):
        
        self.ell = ell
        self.nus = nus
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


class Fitting(data):
    '''
    Class to perform MCMC on the chosen sky parameters
    '''

    def __init__(self):

        with open('fitting_config.yml', "r") as stream:
            self.param_fitting = yaml.safe_load(stream)
        data.__init__(self)
        self.ell, _ = NamasterEll().ell()
        self.sky_parameters = self.param_fitting['SKY_PARAMETERS']
        self.ndim, self.sky_parameters_fitted_names, self.sky_parameters_all_names = self.ndim_and_parameters_names()
        
        #if self.param_fitting['noise_covariance'] is not True:
        reshaped_noise_ps = np.reshape(self.power_spectra_noise, (self.nrec, self.nrec, self.nreal, len(self.ell)))
        self.noise_cov_matrix = np.zeros((self.nrec, self.nrec, len(self.ell), len(self.ell))) 
        for i in range(self.nrec):
            for j in range(self.nrec):
                self.noise_cov_matrix[i, j] = np.cov(reshaped_noise_ps[i,j], rowvar = False)

    def ndim_and_parameters_names(self):
        '''
        Function to create the name list of the parameter(s) that you want to find with the MCMC and to compute the number of these parameters
        
        Return :
            - ndim (int) : number of parameters you want to fit
            - sky_parameters_fitted_names (array) [ndim] : list that contains the names of the fitted parameters
            - sky_parameters_all_names (array) : list that contains the names of all the sky parameters
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

    def dl_to_cl(self, dl):
        cl = np.zeros(self.ell.shape[0])
        for i in range(self.ell.shape[0]):
            cl[i] = dl[i]*(2*np.pi)/(self.ell[i]*(self.ell[i] + 1))
        return cl
    
    def knox_errors(self, clth):
        dcl = (2. / (2 * self.ell + 1) / 0.01 / self.simu_parameters['Spectrum']['dl']) * clth
        return dcl

    def knox_covariance(self, clth):
        dcl = self.knox_errors(clth)
        return np.diag(dcl ** 2)

    def prior(self, x):
        '''
        Function to define priors to help the MCMC convergence

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - (float) : inf if the prior is not respected, 0 otherwise
        '''
        
        for isky_param, sky_param in enumerate(x):
            name = self.sky_parameters_fitted_names[isky_param]

            if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                return - np.inf

        return 0

    def initial_conditions(self):
        '''
        Function to computes the MCMC initial conditions

        Return :
            - p0 (array) [nwalkers, ndim] : array that contains all the initial conditions for the mcmc
        '''

        nwalkers = self.param_fitting['MCMC']['nwalkers']

        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            for j in range(self.ndim):
                name = self.sky_parameters_fitted_names[j]
                p0[i,j] = np.random.random() * self.param_fitting['SKY_PARAMETERS'][name][2] - self.param_fitting['SKY_PARAMETERS'][name][1]

        return p0

    def ptform_uniform(self, u):
        '''
        Function to perform an uniform prior transform for the Nested fitting

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - ptform (array) [ndim] 
        '''

        ptform = []
        cpt = 0
        for iname in self.sky_parameters_all_names:
            if self.param_fitting['SKY_PARAMETERS'][iname][0] is True:
                ptform.append(u[cpt]*self.param_fitting['SKY_PARAMETERS'][iname][2] - self.param_fitting['SKY_PARAMETERS'][iname][1])
                cpt += 1
        return ptform

    def loglikelihood(self, tab):
        '''
        loglikelihood function

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - (float) : loglikelihood function
        '''
        tab_parameters = np.zeros(len(self.param_fitting['SKY_PARAMETERS']))
        cpt = 0        

        for i, iname in enumerate(self.param_fitting['SKY_PARAMETERS']):
            if self.param_fitting['SKY_PARAMETERS'][iname][0] is not True:
                tab_parameters[i] = self.param_fitting['SKY_PARAMETERS'][iname][0]
            else:
                tab_parameters[i] = tab[cpt]
                cpt += 1
        r, Alens, nu0_d, Ad, alphad, betad, deltad = tab_parameters

        # Loglike initialisation + prior
        loglike = self.prior(tab) 

        # Define the sky model & the sample variance associated
        model = 0
        if self.param_fitting['simu']['cmb']:
            model += CMB(self.ell).model_cmb(r, Alens) + np.zeros((self.nrec, self.nrec, len(self.ell)))
            dlth_cmb = CMB(self.ell).model_cmb(r, Alens)
            sample_cov_cmb = self.knox_covariance(dlth_cmb)
        if self.param_fitting['simu']['dust']:
            model += Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d)
            dlth_dust = Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d)
            sample_cov_dust = np.zeros((self.nrec, self.nrec, len(self.ell), len(self.ell)))
            for i in range(self.nrec):
                for j in range(i, self.nrec):
                    sample_cov_dust[i][j] = self.knox_covariance(dlth_dust[i][j])

        # Define the noise model
        if self.param_fitting['simu']['noise'] == False:
            noise_matrix = np.identity(np.shape(self.noise_cov_matrix))
        if self.param_fitting['fitting']['noise_covariance'] is not True:
            noise_matrix = np.zeros(np.shape(self.noise_cov_matrix)) 
            for i in range(self.nrec):
                for j in range(i, self.nrec):
                    for k in range(len(self.ell)):
                        noise_matrix[i][j][k][k] = (self.error_ps_noise[i][j][k])**2
        else :
            noise_matrix = self.noise_cov_matrix

        if self.param_fitting['fitting']['cmb_sample_variance']:
            noise_matrix += sample_cov_cmb
        if self.param_fitting['fitting']['dust_sample_variance']:
            noise_matrix += sample_cov_dust
        
        for i in range(self.nrec):
            for j in range(i, self.nrec):
                self.inv_noise_matrix = np.linalg.pinv(noise_matrix[i][j])
                loglike += - 0.5 * (self.mean_ps_sky[i][j] - model[i][j]).T @ self.inv_noise_matrix @ (self.mean_ps_sky[i][j] - model[i][j])
        return loglike

    def MCMC(self):
        '''
        Funtion to perform the MCMC and save the results
        '''

        # Define the MCMC parameters, initial conditions and ell list
        nwalkers = self.param_fitting['MCMC']['nwalkers']
        mcmc_steps = self.param_fitting['MCMC']['mcmc_steps']
        p0 = self.initial_conditions()
        
        print(self.simu_parameters)
        
        # Start the MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_prob_fn = self.loglikelihood, pool = pool, moves = [(emcee.moves.StretchMove(), self.param_fitting['MCMC']['stretch_move_factor']), (emcee.moves.DESnookerMove(gammas=self.param_fitting['MCMC']['snooker_move_gamma']), 1 - self.param_fitting['MCMC']['stretch_move_factor'])])
            sampler.run_mcmc(p0, mcmc_steps, progress=True)

        samples_flat = sampler.get_chain(flat = True, discard = self.param_fitting['MCMC']['discard'], thin = self.param_fitting['MCMC']['thin'])
        samples = sampler.get_chain()

        # Plot the walkers
        fig, ax = plt.subplots(1, self.ndim, figsize = (15, 5))
        for j in range(self.ndim):
            for i in range(nwalkers):
                ax[j].plot(samples[:, i, j])
                ax[j].set_title(self.sky_parameters_fitted_names[j])

        name = self.param_fitting['data']['name']
        config = self.param_fitting['simu']['qubic_config']
        nrec = self.param_fitting['simu']['nrec']
        n_real = self.param_fitting['data']['n_real']
        convo = self.param_fitting['simu']['convo']
        loglike = 'noise_cov_' + str(self.param_fitting['fitting']['noise_covariance']) + '_sample_var_' + str(self.param_fitting['fitting']['dust_sample_variance'])
        path_plot = f'{name}_{config}_Nrec={nrec}_Loglike={loglike}_Convolution={convo}_plots_MCMC'
        if not os.path.isdir(path_plot):
            os.makedirs(path_plot)
        fig.suptitle(f'Walkers plot - Nreal={n_real} ' + path_plot) 
        plt.savefig(path_plot + f'/walkers_plot_Nreal={n_real}')

        mcmc_values = np.mean(samples_flat, axis=0)
        parameters_values = []
        cpt=0
        for parameter in self.sky_parameters:
            if self.sky_parameters[parameter][0] is True:
                parameters_values.append(mcmc_values[cpt])
                cpt+=1
            else:
                parameters_values.append(self.sky_parameters[parameter][0])

        r, Alens, nu0_d, Ad, alphad, betad, deltad = parameters_values

        # Define the noise matrix
        if self.param_fitting['simu']['cmb']:
            dlth_cmb = CMB(self.ell).model_cmb(r, Alens)
            sample_cov_cmb = self.knox_covariance(dlth_cmb)
        if self.param_fitting['simu']['dust']:
            dlth_dust = Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d)
            sample_cov_dust = np.zeros((self.nrec, self.nrec, len(self.ell), len(self.ell)))
            for i in range(self.nrec):
                for j in range(i, self.nrec):
                    sample_cov_dust[i][j] = self.knox_covariance(dlth_dust[i][j])

        # Define the noise model
        if self.param_fitting['simu']['noise'] == False:
            noise_matrix = np.identity(np.shape(self.noise_cov_matrix))
        if self.param_fitting['fitting']['noise_covariance'] is not True:
            noise_matrix = np.zeros(np.shape(self.noise_cov_matrix)) 
            for i in range(self.nrec):
                for j in range(i, self.nrec):
                    for k in range(len(self.ell)):
                        noise_matrix[i][j][k][k] = (self.error_ps_noise[i][j][k])**2
        else :
            noise_matrix = self.noise_cov_matrix

        if self.param_fitting['fitting']['cmb_sample_variance']:
            noise_matrix += sample_cov_cmb
        if self.param_fitting['fitting']['dust_sample_variance']:
            noise_matrix += sample_cov_dust

        inv_cov_matrix = np.linalg.pinv(noise_matrix)
        error_bar = np.zeros((self.nrec, self.nrec, len(self.ell)))
        for i in range(self.nrec):
            for j in range(i, self.nrec):
                error_bar[i][j] = np.diag(inv_cov_matrix[i][j])

        # Triangle plot
        plt.figure()
        s = MCSamples(samples=samples_flat, names=self.sky_parameters_fitted_names, labels=self.sky_parameters_fitted_names)
        g = plots.get_subplot_plotter(width_inch=10)
        g.triangle_plot([s], filled=True, title_limit=1)
        for ax in g.subplots[:,0]:
            ax.axvline(0, color='gray')

        path_plot_triangle = f'{config}_Nrec={nrec}_plots'
        #fig.suptitle(f'Triangle plot - Nreal={n_real} ' + path_plot) 
        plt.savefig(path_plot + f'/triangle_plot_Nreal={n_real}')

        # Data vs Fit plot
        fig, axes = plt.subplots(self.nrec, self.nrec, figsize = (10,8))
        
        Dl_mcmc = CMB(self.ell).model_cmb(parameters_values[0], parameters_values[1]) + Dust(self.ell, self.nus).model_dust(parameters_values[3], parameters_values[4], parameters_values[5], parameters_values[6], parameters_values[2])
        Dl_test = CMB(self.ell).model_cmb(0, 1) + Dust(self.ell, self.nus).model_dust(13, -0.17, parameters_values[5], parameters_values[6], parameters_values[2])        
        
        for x in range(self.nrec):
            for y in range(x, self.nrec):
                #axes[x,y].plot(self.ell[:5], Dl_test[x][y][:5], label = 'Model test : r=0, Ad=10, alphad=-0.05')
                axes[x,y].plot(self.ell[:5], Dl_mcmc[x][y][:5], label = 'MCMC')
                axes[x,y].plot(self.ell[:5], self.mean_ps_sky[x][y][:5], label = 'Data')
                axes[x,y].legend()
                axes[x,y].set_xlabel('l')
                axes[x,y].set_ylabel('Dl')
                axes[x,y].set_title(f'{int(self.nus[x])} x {int(self.nus[y])}')
        fig.suptitle(f'Power spectra comparison - Nreal={n_real} ' + path_plot)      
        plt.savefig(path_plot + f'/Comparison_plot_Nreal={n_real}')
        
        
        fig, axes = plt.subplots(self.nrec, self.nrec, figsize = (10,8))
        for x in range(self.nrec):
            for y in range(x, self.nrec):
                #axes[x,y].plot(self.ell, Dl_test[x][y], label = 'Model test : r=0, Ad=10, alphad=-0.05')
                axes[x,y].plot(self.ell, Dl_mcmc[x][y], label = 'MCMC')
                axes[x,y].plot(self.ell, self.mean_ps_sky[x][y], label = 'Data')
                axes[x,y].legend()
                axes[x,y].set_xlabel('l')
                axes[x,y].set_ylabel('Dl')
                axes[x,y].set_title(f'{int(self.nus[x])} x {int(self.nus[y])}')
        fig.suptitle(f'Power spectra comparison - Nreal={n_real} ' + path_plot)      
        plt.savefig(path_plot + f'/Comparison_plot_extended_Nreal={n_real}')
        
        
        fig, axes = plt.subplots(self.nrec, self.nrec, figsize = (10,8))
        for x in range(self.nrec):
            for y in range(x, self.nrec):
                #axes[x,y].plot(self.ell, Dl_test[x][y], label = 'Model test : r=0, Ad=10, alphad=-0.05')
                axes[x,y].plot(self.ell, Dl_mcmc[x][y], label = 'MCMC')
                axes[x,y].errorbar(self.ell, self.mean_ps_sky[x][y], error_bar[x][y], label = 'Data')
                axes[x,y].legend()
                axes[x,y].set_xlabel('l')
                axes[x,y].set_ylabel('Dl')
                axes[x,y].set_title(f'{int(self.nus[x])} x {int(self.nus[y])}')
        fig.suptitle(f'Power spectra comparison - Nreal={n_real} ' + path_plot)      
        plt.savefig(path_plot + f'/Comparison_plot_error_extended_Nreal={n_real}')
        

    def nested_fitting(self):
        '''
        Funtion to perform the Nested fitting and save the results
        '''

        nlive = self.param_fitting['NS']['nlive']
        maxiter = self.param_fitting['NS']['maxiter']
        print(self.simu_parameters)
        
        if self.param_fitting['NS']['DynamicNS'] is True:
            print('Dynamic Nested fitting !!!')
            with Pool() as pool:
                sampler_ns = DynamicNestedSampler(self.loglikelihood, self.ptform_uniform, self.ndim, pool = pool, nlive = nlive, queue_size=self.param_fitting['NS']['queue_size'], bound=self.param_fitting['NS']['bound'])
                sampler_ns.run_nested(print_progress=True, maxiter = maxiter)
        else:
            print('Nested fitting !')
            with Pool() as pool:
                sampler_ns = NestedSampler(self.loglikelihood, self.ptform_uniform, self.ndim, pool = pool, nlive = nlive, queue_size=self.param_fitting['NS']['queue_size'], bound=self.param_fitting['NS']['bound'])
                sampler_ns.run_nested(maxiter = maxiter)

        results = sampler_ns.results

        # Plot the traceplots
        fig, axes = dyplot.traceplot(results, show_titles=True, labels = self.sky_parameters_fitted_names,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))

        config = self.param_fitting['simu']['qubic_config']
        nrec = self.param_fitting['simu']['nrec']
        n_real = self.param_fitting['data']['n_real']
        convo = self.param_fitting['simu']['convo']
        name = self.param_fitting['data']['name']
        loglike = 'noise_cov_' + str(self.param_fitting['fitting']['noise_covariance']) + '_sample_var_' + str(self.param_fitting['fitting']['dust_sample_variance'])
        if self.param_fitting['NS']['DynamicNS'] is True:
            path_plot = f'{name}_{config}_Nrec={nrec}_Loglike={loglike}_Convolution={convo}_plots_DynamicNS'
        else:
            path_plot = f'{name}_{config}_Nrec={nrec}_Loglike={loglike}_Convolution={convo}_plots_NS'
        if not os.path.isdir(path_plot):
            os.makedirs(path_plot)
        #plt.title(f'Traceplot - Nreal={n_real} ' + path_plot)
        plt.savefig(path_plot + f'/traceplot_Nreal={n_real}')

        # Runplots
        fig, axes = dyplot.runplot(results)
        #plt.title(f'Runplot - Nreal={n_real} ' + path_plot)
        plt.savefig(path_plot + f'/runplot_Nreal={n_real}')


        # Triangle plot
        fig, axes = plt.subplots(3, 3)
        axes = axes.reshape((3, 3))
        fg, ax = dyplot.cornerplot(results, color='blue', title_fmt = '.4f', show_titles=True, labels = self.sky_parameters_fitted_names,
                           max_n_ticks=3, quantiles=None,
                           fig=(fig, axes[:, :3]))
        #fig.suptitle(f'Triangleplot - Nreal={n_real} ' + path_plot)
        plt.savefig(path_plot + f'/triangle_plot_Nreal={n_real}')

        # Data vs Fit plot
        fig, axes = plt.subplots(self.nrec, self.nrec, figsize = (10,8))
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
        Dl_ns = CMB(self.ell).model_cmb(parameters_values[0], parameters_values[1]) + Dust(self.ell, self.nus).model_dust(parameters_values[3], parameters_values[4], parameters_values[5], parameters_values[6], parameters_values[2])
        Dl_test = CMB(self.ell).model_cmb(0, 1) + Dust(self.ell, self.nus).model_dust(10, -0.15, parameters_values[5], parameters_values[6], parameters_values[2])                 
    
        for x in range(self.nrec):
            for y in range(x, self.nrec):
                axes[x,y].plot(self.ell[:5], Dl_test[x][y][:5], label = 'Model test : r=0, Ad=10, alphad=-0.15')
                axes[x,y].plot(self.ell[:5], Dl_ns[x][y][:5], label = 'NS')
                axes[x,y].errorbar(self.ell[:5], self.mean_ps_sky[x][y][:5], self.error_ps_sky[x][y][:5], label = 'Data')
                axes[x,y].legend()
                axes[x,y].set_xlabel('l')
                axes[x,y].set_ylabel('Dl')
                axes[x,y].set_title(f'{int(self.nus[x])} x {int(self.nus[y])}')
        fig.suptitle(f'Power spectra comparison - Nreal={n_real} ' + path_plot) 
        plt.savefig(path_plot + f'/Comparison_plot_Nreal={n_real}')

        fig, axes = plt.subplots(self.nrec, self.nrec, figsize = (10,8))
        for x in range(self.nrec):
            for y in range(x, self.nrec):
                axes[x,y].plot(self.ell, Dl_test[x][y], label = 'Model test : r=0, Ad=10, alphad=-0.15')
                axes[x,y].plot(self.ell, Dl_ns[x][y], label = 'NS')
                axes[x,y].errorbar(self.ell, self.mean_ps_sky[x][y], self.error_ps_sky[x][y], label = 'Data')
                axes[x,y].legend()
                axes[x,y].set_xlabel('l')
                axes[x,y].set_ylabel('Dl')
                axes[x,y].set_title(f'{int(self.nus[x])} x {int(self.nus[y])}')
        fig.suptitle(f'Power spectra comparison - Nreal={n_real} ' + path_plot) 
        plt.savefig(path_plot + f'/Comparison_plot_extended_Nreal={n_real}')


class Nestedfitting(data):
    '''
    Class to perform Nested fitting in our sky parameters
    '''

    def __init__(self):

        with open('fitting_config.yml', "r") as stream:
            self.param_fitting = yaml.safe_load(stream)
        data.__init__(self)
        self.ell, _ = NamasterEll().ell()
        self.sky_parameters = self.param_fitting['SKY_PARAMETERS']
        self.ndim, self.sky_parameters_fitted_names, self.sky_parameters_all_names = self.ndim_and_parameters_names()
        
        if self.param_fitting['Loglike'] == 'cov' :
            reshaped_noise_ps = np.reshape(self.power_spectra_noise, (self.nrec, self.nrec, self.nreal, 16))
            self.noise_cov_matrix = np.zeros((self.nrec, self.nrec, 16, 16)) 
            for i in range(self.nrec):
                for j in range(self.nrec):
                    self.noise_cov_matrix[i, j] = np.cov(reshaped_noise_ps[i,j], rowvar = False)

    def ndim_and_parameters_names(self):
        '''
        Function to create the name list of the parameter(s) that you want to find with the MCMC and to compute the number of these parameters
        
        Return :
            - ndim (int) : number of parameters you want to fit
            - sky_parameters_fitted_names (array) [ndim] : list that contains the names of the fitted parameters
            - sky_parameters_all_names (array) : list that contains the names of all the sky parameters
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

        Return :
            - p0 (array) [nwalkers, ndim] : array that contains all the initial conditions for the mcmc
        '''

        nwalkers = self.param_fitting['MCMC']['nwalkers']

        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            for j in range(self.ndim):
                name = self.sky_parameters_fitted_names[j]
                p0[i,j] = np.random.random() * self.param_fitting['SKY_PARAMETERS'][name][2] - self.param_fitting['SKY_PARAMETERS'][name][1]

        return p0

    def prior(self, x):
        '''
        Function to define priors to help the MCMC convergence

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - (float) : inf if the prior is not respected, 0 otherwise
        '''
        
        for isky_param, sky_param in enumerate(x):
            name = self.sky_parameters_fitted_names[isky_param]

            if sky_param < self.sky_parameters[name][3] or sky_param > self.sky_parameters[name][4]:
                return - np.inf

        return 0

    def loglikelihood(self, tab):
        '''
        loglikelihood function

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - (float) : loglikelihood function
        '''
        tab_parameters = np.zeros(len(self.param_fitting['SKY_PARAMETERS']))
        cpt = 0        

        for i, iname in enumerate(self.param_fitting['SKY_PARAMETERS']):
            if self.param_fitting['SKY_PARAMETERS'][iname][0] is not True:
                tab_parameters[i] = self.param_fitting['SKY_PARAMETERS'][iname][0]
            else:
                tab_parameters[i] = tab[cpt]
                cpt += 1

        r, Alens, nu0_d, Ad, alphad, betad, deltad = tab_parameters
        if self.param_fitting['simu']['noise'] == False:
            loglike = self.prior(tab) 
            for i in range(self.nrec):
                for j in range(self.nrec):
                    loglike += - 0.5 * (self.mean_ps_sky[i][j] - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d))[i][j]).T @ (self.mean_ps_sky[i][j] - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d))[i][j])
                    #loglike = self.prior(tab) - 0.5 * np.sum(((self.mean_ps_sky - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d)))**2)
            return loglike

        if self.param_fitting['simu']['name'] == 'CMB':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_ps_sky - CMB(self.ell).model_cmb(r, Alens))/(self.error_ps_noise))**2)
        if self.param_fitting['simu']['name'] == 'Dust':
            return self.prior(tab) - 0.5 * np.sum(((self.mean_ps_sky - Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d))/(self.error_ps_noise))**2)
        if self.param_fitting['simu']['name'] == 'Sky':
            loglike = self.prior(tab) 
            if self.param_fitting['Loglike'] == 'cov':
                dlth = CMB(self.ell).model_cmb(r, Alens)
                sample_cov = CMB(self.ell).knox_covariance(dlth)
                cov_matrix = self.noise_cov_matrix + sample_cov
                inv_cov_matrix = np.linalg.pinv(cov_matrix)
                for i in range(self.nrec):
                    for j in range(self.nrec):
                        loglike += - 0.5 * (self.mean_ps_sky[i][j] - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d))[i][j]).T @ inv_cov_matrix[i][j] @ (self.mean_ps_sky[i][j] - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d))[i][j])
                return loglike
            else:
                loglike += - 0.5 * np.sum(((self.mean_ps_sky - (CMB(self.ell).model_cmb(r, Alens) + Dust(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d)))/(self.error_ps_noise))**2)
            return loglike

    def ptform_uniform(self, u):
        '''
        Function to perform an uniform prior transform for the Nested fitting

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - ptform (array) [ndim] 
        '''

        ptform = []
        cpt = 0
        for iname in self.sky_parameters_all_names:
            if self.param_fitting['SKY_PARAMETERS'][iname][0] is True:
                ptform.append(u[cpt]*self.param_fitting['SKY_PARAMETERS'][iname][2] - self.param_fitting['SKY_PARAMETERS'][iname][1])
                cpt += 1
        return ptform

    def __call__(self):
        '''
        Funtion to perform the Nested fitting and save the results
        '''

        nlive = self.param_fitting['NS']['nlive']
        maxiter = self.param_fitting['NS']['maxiter']
        print(self.simu_parameters)
        
        if self.param_fitting['NS']['DynamicNS'] is True:
            print('Dynamic Nested fitting !!!')
            with Pool() as pool:
                sampler_ns = DynamicNestedSampler(self.loglikelihood, self.ptform_uniform, self.ndim, pool = pool, nlive = nlive, queue_size=self.param_fitting['NS']['queue_size'], bound=self.param_fitting['NS']['bound'])
                sampler_ns.run_nested(print_progress=True, maxiter = maxiter)
        else:
            print('Nested fitting !')
            with Pool() as pool:
                sampler_ns = NestedSampler(self.loglikelihood, self.ptform_uniform, self.ndim, pool = pool, nlive = nlive, queue_size=self.param_fitting['NS']['queue_size'], bound=self.param_fitting['NS']['bound'])
                sampler_ns.run_nested(maxiter = maxiter)

        results = sampler_ns.results

        # Plot the traceplots
        fig, axes = dyplot.traceplot(results, show_titles=True, labels = self.sky_parameters_fitted_names,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))

        config = self.param_fitting['simu']['qubic_config']
        nrec = self.param_fitting['simu']['nrec']
        n_real = self.param_fitting['data']['n_real']
        convo = self.param_fitting['simu']['convo']
        name = self.param_fitting['data']['name']
        if self.param_fitting['Loglike'] == "cov":
            loglike = 'Cov'
        else :
            loglike = 'Diag'
        if self.param_fitting['NS']['DynamicNS'] is True:
            path_plot = f'{name}_{config}_Nrec={nrec}_Loglike={loglike}_Convolution={convo}_plots_DynamicNS'
        else:
            path_plot = f'{name}_{config}_Nrec={nrec}_Loglike={loglike}_Convolution={convo}_plots_NS'
        if not os.path.isdir(path_plot):
            os.makedirs(path_plot)
        #plt.title(f'Traceplot - Nreal={n_real} ' + path_plot)
        plt.savefig(path_plot + f'/traceplot_Nreal={n_real}')

        # Runplots
        fig, axes = dyplot.runplot(results)
        #plt.title(f'Runplot - Nreal={n_real} ' + path_plot)
        plt.savefig(path_plot + f'/runplot_Nreal={n_real}')


        # Triangle plot
        fig, axes = plt.subplots(3, 3)
        axes = axes.reshape((3, 3))
        fg, ax = dyplot.cornerplot(results, color='blue', title_fmt = '.4f', show_titles=True, labels = self.sky_parameters_fitted_names,
                           max_n_ticks=3, quantiles=None,
                           fig=(fig, axes[:, :3]))
        #fig.suptitle(f'Triangleplot - Nreal={n_real} ' + path_plot)
        plt.savefig(path_plot + f'/triangle_plot_Nreal={n_real}')

        # Data vs Fit plot
        fig, axes = plt.subplots(self.nrec, self.nrec, figsize = (10,8))
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
        Dl_ns = CMB(self.ell).model_cmb(parameters_values[0], parameters_values[1]) + Dust(self.ell, self.nus).model_dust(parameters_values[3], parameters_values[4], parameters_values[5], parameters_values[6], parameters_values[2])
        Dl_test = CMB(self.ell).model_cmb(0, 1) + Dust(self.ell, self.nus).model_dust(10, -0.15, parameters_values[5], parameters_values[6], parameters_values[2])                 
    
        for x in range(self.nrec):
            for y in range(self.nrec):
                axes[x,y].plot(self.ell[:5], Dl_test[x][y][:5], label = 'Model test : r=0, Ad=10, alphad=-0.15')
                axes[x,y].plot(self.ell[:5], Dl_ns[x][y][:5], label = 'NS')
                axes[x,y].errorbar(self.ell[:5], self.mean_ps_sky[x][y][:5], self.error_ps_sky[x][y][:5], label = 'Data')
                axes[x,y].legend()
                axes[x,y].set_xlabel('l')
                axes[x,y].set_ylabel('Dl')
                axes[x,y].set_title(f'{int(self.nus[x])} x {int(self.nus[y])}')
        fig.suptitle(f'Power spectra comparison - Nreal={n_real} ' + path_plot) 
        plt.savefig(path_plot + f'/Comparison_plot_Nreal={n_real}')

        fig, axes = plt.subplots(self.nrec, self.nrec, figsize = (10,8))
        for x in range(self.nrec):
            for y in range(self.nrec):
                axes[x,y].plot(self.ell, Dl_test[x][y], label = 'Model test : r=0, Ad=10, alphad=-0.15')
                axes[x,y].plot(self.ell, Dl_ns[x][y], label = 'NS')
                axes[x,y].errorbar(self.ell, self.mean_ps_sky[x][y], self.error_ps_sky[x][y], label = 'Data')
                axes[x,y].legend()
                axes[x,y].set_xlabel('l')
                axes[x,y].set_ylabel('Dl')
                axes[x,y].set_title(f'{int(self.nus[x])} x {int(self.nus[y])}')
        fig.suptitle(f'Power spectra comparison - Nreal={n_real} ' + path_plot) 
        plt.savefig(path_plot + f'/Comparison_plot_extended_Nreal={n_real}')

with open('fitting_config.yml', "r") as stream:
    param = yaml.safe_load(stream)
print('fitting Parameters', param)

if param['fitting']['Method'] == 'MCMC':
    print("Chosen method = MCMC")
    Fitting().MCMC()
elif param['fitting']['Method'] == 'NS':
    print("Chosen method = Nested fitting")
    Fitting().nested_fitting()
else:
    print('Wrong fitting method')

print("Fitting done")













