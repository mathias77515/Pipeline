#### General packages
import pickle
import os
import os.path as op
import sys
import numpy as np
import healpy as hp
import emcee
import yaml
from schwimmbad import MPIPool
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.gridspec import *

sys.path.append('/pbs/home/m/mregnier/sps1/Pipeline')

#### QUBIC packages
import fgb.component_model as c
import fgb.mixing_matrix as mm
from pyoperators import *

comm = MPI.COMM_WORLD
def _Dl2Cl(ell, Dl):
    _f = ell * (ell + 1) / (2 * np.pi)
    return Dl / _f
def _Cl2BK(ell, Cl):
    return 100 * ell * Cl / (2 * np.pi)

class data:
    '''
    Class to extract of the power spectra computed with spectrum.py and to compute useful things
    '''

    def __init__(self):

        with open('fit_config_SO.yaml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.is_cov_matrix = self.params['fitting']['noise_covariance_matrix']
        self.is_samp_cmb = self.params['fitting']['cmb_sample_variance']
        self.is_samp_dust = self.params['fitting']['dust_sample_variance']
        
        self.path_repository = self.params['data']['path']
        self.path_spectra = self.path_repository + self.params['data']['foldername']
        self.path_fit = self.path_repository + 'fit' + f"_{self.params['data']['filename']}"

        ### Create fit folder
        if comm.Get_rank() == 0:
            if not os.path.isdir(self.path_fit):
                os.makedirs(self.path_fit)
                
        ### Read datasets
        self.power_spectra_sky, self.power_spectra_noise, self.simu_parameters, self.coverage, self.nus, self.ell = self.import_power_spectra(self.path_spectra)
        self._f = self.ell * (self.ell + 1) / (2 * np.pi)
        self.fsub = self.simu_parameters['QUBIC']['fsub']
        self.nrec = self.simu_parameters['QUBIC']['nrec']
        self.nsub = self.fsub * self.nrec
        self.nreal = self.params['data']['n_real']
        
        ### Select bandpowers for fitting
        bp_to_rm = self.select_bandpower()
        self.nfreq = len(self.nus)
        self.nspecs = (self.nfreq * (self.nfreq + 1)) // 2
    
        ### Remove bandpowers not selected
        self.power_spectra_sky = np.delete(self.power_spectra_sky, bp_to_rm, 1)
        self.power_spectra_sky = np.delete(self.power_spectra_sky, bp_to_rm, 2)
        self.power_spectra_noise = np.delete(self.power_spectra_noise, bp_to_rm, 1)
        self.power_spectra_noise = np.delete(self.power_spectra_noise, bp_to_rm, 2)

        ### Average and standard deviation from realizations
        self.mean_ps_sky, self.error_ps_sky = self.compute_mean_std(self.power_spectra_sky)
        self.mean_ps_noise, self.error_ps_noise = self.compute_mean_std(self.power_spectra_noise)
        

        if self.params['simu']['noise'] is True:
            self.mean_ps_sky = self.spectra_noise_correction(self.mean_ps_sky, self.mean_ps_noise)

        self.ps_sky_reshape = np.zeros((self.params['data']['n_real'], self.nspecs, len(self.ell)))
        self.ps_noise_reshape = np.zeros((self.params['data']['n_real'], self.nspecs, len(self.ell))) 
        self.mean_ps_sky_reshape = np.zeros((self.nspecs, len(self.ell)))
        self.mean_ps_noise_reshape = np.zeros((self.nspecs, len(self.ell))) 
        
        for ireal in range(self.params['data']['n_real']):
            k=0
            for i in range(self.nfreq):
                for j in range(i, self.nfreq):
                    self.ps_sky_reshape[ireal, k] = self.power_spectra_sky[ireal][i, j].copy()
                    self.ps_noise_reshape[ireal, k] = self.power_spectra_noise[ireal][i, j].copy()
                    k+=1
        k=0
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                self.mean_ps_sky_reshape[k] = self.mean_ps_sky[i, j].copy()
                self.mean_ps_noise_reshape[k] = self.mean_ps_noise[i, j].copy()
                k+=1
        
    def select_bandpower(self):
        '''
        Function to remove some bamdpowers if they are not selected.
        
        Return : 
            - list containing the indices for removed bandpowers.
        '''
        k=0
        bp_to_rm = []
        for ii, i in enumerate(self.nus):
            if ii < self.params['NUS']['qubic'][1]:
                if self.params['NUS']['qubic'][0]:
                    k += (self.params['NUS']['qubic'][1])
                else:
                    bp_to_rm += [ii]
                    k+=1
                
            else:
                if self.params['NUS'][f'{i:.0f}GHz'] is False:
                    bp_to_rm += [ii]
        #print(bp_to_rm)
        self.nus = np.delete(self.nus, bp_to_rm, 0)
        return bp_to_rm
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
        for i in range(self.params['data']['n_real']):
            #if comm.Get_rank() == 0:
                #print(f"======== Importing power spectrum {i+1} / {self.params['data']['n_real']} ==========")
            ps = pickle.load(open(path + '/' + names[i], 'rb'))
            power_spectra_sky.append(ps['Dls'][:, :, :self.params['nbins']])
            power_spectra_noise.append(ps['Nl'][:, :, :self.params['nbins']])
        return power_spectra_sky, power_spectra_noise, ps['parameters'], ps['coverage'], ps['nus'], ps['ell'][:self.params['nbins']]
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
                mean_data[i, j, :] -= mean_noise[i, j, :]
        return mean_data

class CMB:
    '''
    Class to define the CMB model
    '''

    def __init__(self, ell, nus):
        
        self.nus = nus
        self.ell = ell
        self.nfreq = len(self.nus)
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
        models = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                models[i, j] = self.cl_to_dl(self.get_pw_from_planck(r, Alens))
        return models
class Foreground:
    '''
    Class to define Dust and Synchrotron models
    '''

    def __init__(self, ell, nus):
        
        self.ell = ell
        self.nus = nus
        self.nfreq = len(self.nus)

    def scale_dust(self, nu, nu0_d, betad, temp=20):
        '''
        Function to compute the dust mixing matrix element, depending on the frequency
        '''

        comp = c.Dust(nu0 = nu0_d, temp=temp, beta_d = betad)
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]
        #print(nu, A)
        #stop
        return A
    def scale_sync(self, nu, nu0_s, betas):
        '''
        Function to compute the dust mixing matrix element, depending on the frequency
        '''

        comp = c.Synchrotron(nu0 = nu0_s, beta_pl = betas)
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]

        return A
    def model_dust_frequency(self, A, alpha, delta, fnu1, fnu2):
        '''
        Function to define the Dust model for two frequencies
        '''

        return abs(A) * delta * fnu1 * fnu2 * (self.ell/80)**alpha
    def model_sync_frequency(self, A, alpha, fnu1, fnu2):
        '''
        Function to define the Dust model for two
        '''

        return abs(A) * fnu1 * fnu2 * (self.ell/80)**alpha
    def model_dustsync_corr(self, Ad, As, alphad, alphas, fnu1d, fnu2d, fnu1s, fnu2s, eps):
        return eps * np.sqrt(abs(As) * abs(Ad)) * (fnu1d * fnu2s + fnu2d * fnu1s) * (self.ell / 80)**((alphad + alphas)/2)
    def model_dust(self, Ad, alphad, betad, deltad, nu0_d):
        '''
        Function defining the Dust model for all frequencies, depending on Ad, alphad, betad, deltad & nu0_d
        '''
        
        models = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                if i == j:
                    #fnud = self.scale_dust(self.nus[i], nu0_d, betad)
                    fnud = self.scale_dust(self.nus[i], nu0_d, betad)
                    models[i, j] += self.model_dust_frequency(Ad, alphad, deltad, fnud, fnud)
                else:
                    #fnu1d = self.scale_dust(self.nus[i], nu0_d, betad)
                    #fnu2d = self.scale_dust(self.nus[j], nu0_d, betad)
                    fnu1d = self.scale_dust(self.nus[i], nu0_d, betad)
                    fnu2d = self.scale_dust(self.nus[j], nu0_d, betad)
                    models[i, j] += self.model_dust_frequency(Ad, alphad, deltad, fnu1d, fnu2d)
        
        return models
    def model_sync(self, As, alphas, betas, nu0_s):
        '''
        Function defining the Synchrotron model for all frequencies
        '''

        models = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        for i in range(self.nfreq):
            for j in range(self.nfreq):
                if i == j:
                    fnus = self.scale_sync(self.nus[i], nu0_s, betas)
                    models[i, j] += self.model_sync_frequency(As, alphas, fnus, fnus)
                else:
                    fnu1s = self.scale_sync(self.nus[i], nu0_s, betas)
                    fnu2s = self.scale_sync(self.nus[j], nu0_s, betas)
                    models[i, j] += self.model_sync_frequency(As, alphas, fnu1s, fnu2s)
        return models 
    def model_corr_dust_sync(self, Ad, alphad, betad, nu0_d, As, alphas, betas, nu0_s, eps):
        '''
        Function defining the correlation model between Dust & Synchrotron for all frequencies
        '''

        models = np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                if i == j:
                    fnud = self.scale_dust(self.nus[i], nu0_d, betad)
                    fnus = self.scale_sync(self.nus[i], nu0_s, betas)
                    models[i, j] += self.model_dustsync_corr(Ad, As, alphad, alphas, fnud, fnud, fnus, fnus, eps)
                else:
                    fnu1d = self.scale_dust(self.nus[i], nu0_d, betad)
                    fnu2d = self.scale_dust(self.nus[j], nu0_d, betad)
                    fnu1s = self.scale_sync(self.nus[i], nu0_s, betas)
                    fnu2s = self.scale_sync(self.nus[j], nu0_s, betas)
                    models[i, j] += self.model_dustsync_corr(Ad, As, alphad, alphas, fnu1d, fnu2d, fnu1s, fnu2s, eps)
        return models


class Fitting(data):
    '''
    Class to perform MCMC on the chosen sky parameters
    '''

    def __init__(self):
        
        if comm.Get_rank() == 0:
            print('\n=========== Fitting ===========\n')

        data.__init__(self)
        self.sky_parameters = self.params['SKY_PARAMETERS']
        self.ndim, self.sky_parameters_fitted_names, self.sky_parameters_all_names = self.ndim_and_parameters_names()
        
        self.noise_cov_matrix = np.zeros((self.nspecs, len(self.ell), len(self.ell), self.nspecs)) 
        #self.sample_cov_matrix = np.zeros((self.nspecs, len(self.ell), len(self.ell), self.nspecs)) 
        
        #print(np.diag(np.cov(self.ps_noise_reshape[:, 0, :], self.ps_noise_reshape[:, 0, :], rowvar = False)[:16, :16]))
        #print(np.diag(np.cov(self.ps_noise_reshape[:, 0, :], self.ps_noise_reshape[:, 1, :], rowvar = False)[:16, 16:32]))
        samples = self.ps_noise_reshape.reshape((self.ps_noise_reshape.shape[0], self.nspecs*len(self.ell)))
        cov = np.cov(samples, rowvar=False)
        #print(cov.shape)
        self.noise_cov_matrix = cov.reshape((self.nspecs*len(self.ell), self.nspecs*len(self.ell)))
        self.sample_cov_matrix = self._fill_sample_variance(self.mean_ps_sky).reshape((self.nspecs*len(self.ell), self.nspecs*len(self.ell)))
        self.covariance = self.noise_cov_matrix + self.sample_cov_matrix
        self.inv_cov = np.linalg.pinv(self.covariance)
        k=0
        #for i in range(self.nfreq):
        #    for j in range(i, self.nfreq):
        #        print(i, j, ' ', i*len(self.ell), (i+1)*len(self.ell), j*len(self.ell), (j+1)*len(self.ell))
        #        #self.noise_cov_matrix[k] = np.cov(self.power_spectra_noise[:, i, j, :], rowvar = False)
        #        k+=1
        #stop

        self.cmb = CMB(self.ell, self.nus)
        self.foregrounds = Foreground(self.ell, self.nus)
        model = self.cmb.model_cmb(0, 1) + self.foregrounds.model_dust(0, -0.17, 1.54, 1, 353)
        
        self._get_Dl_plot(self.nus, self.ell, self.mean_ps_sky, self.error_ps_noise, model, nbins=self.params['nbins'], nrec=self.nrec)

        
    def _get_Dl_plot(self, nus, ell, Dl, Dl_err, ymodel, nbins=8, nrec=2):
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(len(nus), len(nus), figure=fig)

        k1=0
        kp=0
        for i in range(len(nus)):
            for j in range(i, len(nus)):
                ax = fig.add_subplot(gs[i, j])
            
                if k1 ==0:
                    ax.plot(ell[:nbins], _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], ymodel[i, j, :nbins])), '--r', label=f'r + Dust + noise')
                else:
                    ax.plot(ell[:nbins], _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], ymodel[i, j, :nbins])), '--r')

            
                ax.patch.set_alpha(0.3)
            
                ax.annotate(f'{nus[i]:.0f}x{nus[j]:.0f}', xy=(0.1, 0.9), xycoords='axes fraction', color='black', weight="bold")
                if i < nrec and j < nrec :
                    ax.set_facecolor("blue")
                    if k1 == 0:
                        ax.errorbar(ell[:nbins], _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])),#Dls_mean[kp] - Nl_mean[kp], 
                            yerr=_Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])), 
                            capsize=5, color='darkblue', fmt='o', label=r'$\mathcal{D}_{\ell}^{\nu_1 \times \nu_2}$')
                    
                    else:
                        ax.errorbar(ell[:nbins], _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])),#Dls_mean[kp] - Nl_mean[kp], 
                            yerr=_Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])), 
                            capsize=5, color='darkblue', fmt='o')
                elif i < nrec and j >= nrec :
                    ax.set_facecolor("skyblue")
                    ax.errorbar(ell[:nbins], _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])),#Dls_mean[kp] - Nl_mean[kp], 
                         yerr=_Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])), 
                         capsize=5, color='blue', fmt='o')
                else:
                    ax.set_facecolor("green")
                    ax.errorbar(ell[:nbins], _Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl[i, j, :nbins])),#Dls_mean[kp] - Nl_mean[kp], 
                            yerr=_Cl2BK(ell[:nbins], _Dl2Cl(ell[:nbins], Dl_err[i, j, :nbins])), 
                            capsize=5, color='darkgreen', fmt='o')
                #ax[i, j].set_title(f'{data["nus"][i]:.0f}x{data["nus"][j]:.0f}')
                kp+=1
            else:
                pass#ax.axis('off')
            k1+=1

        plt.tight_layout()
        plt.savefig('Dls.png')
        plt.close()
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
    def knox_errors(self, dlth):
        dcl = np.sqrt(2 / ((2 * self.ell + 1) * 0.01 * self.simu_parameters['Spectrum']['dl'])) * dlth
        return dcl
    def knox_covariance(self, dlth):
        dcl = self.knox_errors(dlth)
        return np.eye(len(self.ell)) * dcl**2
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

        nwalkers = self.params['MCMC']['nwalkers']

        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            for j in range(self.ndim):
                name = self.sky_parameters_fitted_names[j]
                p0[i,j] = np.random.random() * self.params['SKY_PARAMETERS'][name][2] + self.params['SKY_PARAMETERS'][name][1]

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
            if self.params['SKY_PARAMETERS'][iname][0] is True:
                ptform.append(u[cpt]*self.params['SKY_PARAMETERS'][iname][2] - self.params['SKY_PARAMETERS'][iname][1])
                cpt += 1
        return ptform
    def _fill_sample_variance(self, bandpower):
        
        indices_tr = np.triu_indices(len(self.nus))
        matrix = np.zeros((self.nspecs, len(self.ell), self.nspecs, len(self.ell)))
        factor_modecount = 1/((2*self.ell+1) * 0.015 * self.simu_parameters['Spectrum']['dl'])
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                covar = (bandpower[i1, j1, :] * bandpower[i2, j2, :] + bandpower[i1, j2, :] * bandpower[i2, j1, :]) * factor_modecount
                matrix[ii, :, jj, :] = np.diag(covar)
        return matrix
    def _reshape_spectra(self, bandpower):
        
        bandpower_reshaped = np.zeros((self.nspecs, len(self.ell)))
        k=0
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                bandpower_reshaped[k] = bandpower[i, j].copy()
                k+=1
        return bandpower_reshaped
    def loglikelihood(self, tab):
        
        '''
        loglikelihood function

        Argument :
            - x (array) [ndim] : array that contains the numbers randomly generated by the mcmc

        Return :
            - (float) : loglikelihood function
        '''
        tab_parameters = np.zeros(len(self.params['SKY_PARAMETERS']))
        cpt = 0        

        for i, iname in enumerate(self.params['SKY_PARAMETERS']):
            if self.params['SKY_PARAMETERS'][iname][0] is not True:
                tab_parameters[i] = self.params['SKY_PARAMETERS'][iname][0]
            else:
                tab_parameters[i] = tab[cpt]
                cpt += 1
        r, Alens, nu0_d, Ad, alphad, betad, deltad, nu0_s, As, alphas, betas, eps = tab_parameters
        
        # Loglike initialisation + prior
        loglike = self.prior(tab) 

        # Define the sky model & the sample variance associated
        model = self.cmb.model_cmb(r, Alens)# + np.zeros((self.nfreq, self.nfreq, len(self.ell)))
        model += self.foregrounds.model_dust(Ad, alphad, betad, deltad, nu0_d)
        
        
        #print(f'Sample variance : ', np.diag(self.sample_cov_matrix[0, :, :, 0]))
        #print(f'Noise variance : ', np.diag(self.noise_cov_matrix.reshape((self.nspecs, len(self.ell), self.nspecs, len(self.ell)))[0, :, 0, :]))
        #sample_cov_matrix = sample_cov_matrix.reshape((self.nspecs*len(self.ell), self.nspecs*len(self.ell)))
        model = self._reshape_spectra(model)
        
        #covariance = self.noise_cov_matrix + sample_cov_matrix
        _r = model - (self.mean_ps_sky_reshape)
        _r = _r.reshape((self.nspecs*len(self.ell)))
        #stop
        loglike -= 0.5 * (_r.T @ self.inv_cov @ _r)
        #stop
        return loglike
    def save_data(self, name, d):

        """
        
        Method to save data using pickle convention.
        
        """
        
        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run(self):
        '''
        Funtion to perform the MCMC and save the results
        '''

        # Define the MCMC parameters, initial conditions and ell list
        nwalkers = self.params['MCMC']['nwalkers']
        mcmc_steps = self.params['MCMC']['mcmc_steps']
        p0 = self.initial_conditions()
        
        # Start the MCMC
        with MPIPool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_prob_fn = self.loglikelihood, pool=pool)#,
            #moves = [(emcee.moves.StretchMove(), self.params['MCMC']['stretch_move_factor']), 
            #         (emcee.moves.DESnookerMove(gammas=self.params['MCMC']['snooker_move_gamma']), 
            #          1 - self.params['MCMC']['stretch_move_factor'])])
            sampler.run_mcmc(p0, mcmc_steps, progress=True)

        self.samples_flat = sampler.get_chain(flat = True, discard = self.params['MCMC']['discard'], thin = self.params['MCMC']['thin'])
        self.samples = sampler.get_chain()

        name = []
        for inu, i in enumerate(self.params['NUS']):
            if inu == 0 and self.params['NUS'][str(i)][0]:
                name += ['QUBIC']
            else:
                if self.params['NUS'][str(i)]:
                    name += [str(i)]

        name = '_'.join(name)
        self.save_data(self.path_fit + f'/fit_dict_{name}.pkl', {'nus':self.nus,
                              'ell':self.ell,
                              'samples': self.samples,
                              'samples_flat': self.samples_flat,
                              'fitted_parameters_names':self.sky_parameters_fitted_names,
                              'parameters': self.params,
                              'Dls' : self.power_spectra_sky,
                              'Nls' : self.power_spectra_noise,
                              'simulation_parameters': self.simu_parameters})        
        print("Fitting done !!!")


fit = Fitting()
fit.run()

if comm.Get_rank() == 0:
    
    print()
    print(f'Average : {np.mean(fit.samples_flat, axis=0)}')
    print(f'Error : {np.std(fit.samples_flat, axis=0)}')
    print()
    
    plt.figure()
    print(fit.samples.shape)
    for i in range(fit.samples.shape[2]):
        plt.subplot(fit.samples.shape[2], 1, i+1)
        plt.plot(fit.samples[:, :, i], '-k', alpha=0.1)
    
    plt.savefig('chains.png')
    plt.close()
    
    












