#### General packages
import pickle
import os
import os.path as op
import sys
import numpy as np
import healpy as hp
import emcee
import yaml
from multiprocessing import Pool

sys.path.append('/pbs/home/t/tlaclave/sps/Pipeline')

#### QUBIC packages
import fgb.component_model as c
import fgb.mixing_matrix as mm

class data:
    '''
    Class to extract of the power spectra computed with spectrum.py and to compute useful things
    '''

    def __init__(self):

        with open('fit_config.yaml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.is_cov_matrix = self.params['fitting']['noise_covariance_matrix']
        self.is_samp_cmb = self.params['fitting']['cmb_sample_variance']
        self.is_samp_dust = self.params['fitting']['dust_sample_variance']

        self.path_repository = self.params['data']['path']
        self.path_spectra = self.path_repository + 'spectrum'
        self.path_fit = self.path_repository + 'fit'
        if not os.path.isdir(self.path_fit):
            os.makedirs(self.path_fit)
        self.power_spectra_sky, self.power_spectra_noise, self.simu_parameters, self.coverage, self.nus, self.ell = self.import_power_spectra(self.path_spectra)
        self.fsub = self.simu_parameters['QUBIC']['fsub']
        self.nrec = self.simu_parameters['QUBIC']['nrec']
        self.nsub = self.fsub * self.nrec
        self.nfreq = len(self.nus)
        self.nreal = self.params['data']['n_real']

        self.mean_ps_sky, self.error_ps_sky = self.compute_mean_std(self.power_spectra_sky)
        self.mean_ps_noise, self.error_ps_noise = self.compute_mean_std(self.power_spectra_noise)
        if self.params['simu']['noise'] is True:
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
        for i in range(self.params['data']['n_real']):
            ps = pickle.load(open(path + '/' + names[i], 'rb'))
            power_spectra_sky.append(ps['Dls'])
            power_spectra_noise.append(ps['Nl'])
        return power_spectra_sky, power_spectra_noise, ps['parameters'], ps['coverage'], ps['nus'], ps['ell']

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


class Foreground:
    '''
    Class to define Dust and Synchrotron models
    '''

    def __init__(self, ell, nus):
        
        self.ell = ell
        self.nus = nus
        self.nfreq = len(self.nus)
        #self.nspec = int(self.nfreq * (self.nfreq + 1) / 2)
    def scale_dust(self, nu, nu0_d, betad, temp=20):
        '''
        Function to compute the dust mixing matrix element, depending on the frequency
        '''

        comp = c.Dust(nu0 = nu0_d, temp=temp, beta_d = betad)
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]

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
            for j in range(self.nfreq):
                if i == j:
                    fnud = self.scale_dust(self.nus[i], nu0_d, betad)
                    models[i, j] = self.model_dust_frequency(Ad, alphad, deltad, fnud, fnud)
                else:
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
            for j in range(self.nfreq):
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

        print('\n=========== Fitting ===========\n')

        data.__init__(self)
        self.sky_parameters = self.params['SKY_PARAMETERS']
        self.ndim, self.sky_parameters_fitted_names, self.sky_parameters_all_names = self.ndim_and_parameters_names()
        
        #if self.params['noise_covariance'] is not True:
        reshaped_noise_ps = np.reshape(self.power_spectra_noise, (self.nfreq, self.nfreq, self.nreal, len(self.ell)))
        self.noise_cov_matrix = np.zeros((self.nfreq, self.nfreq, len(self.ell), len(self.ell))) 
        for i in range(self.nfreq):
            for j in range(self.nfreq):
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
    
    def knox_errors(self, dlth):
        dcl = (2. / (2 * self.ell + 1) / 0.01 / self.simu_parameters['Spectrum']['dl']) * dlth
        return dcl

    def knox_covariance(self, dlth):
        dcl = self.knox_errors(dlth)
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

        nwalkers = self.params['MCMC']['nwalkers']

        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            for j in range(self.ndim):
                name = self.sky_parameters_fitted_names[j]
                p0[i,j] = np.random.random() * self.params['SKY_PARAMETERS'][name][2] - self.params['SKY_PARAMETERS'][name][1]

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
        model = 0
        if self.params['simu']['cmb']:
            dlth_cmb = CMB(self.ell).model_cmb(r, Alens)
            model += dlth_cmb + np.zeros((self.nfreq, self.nfreq, len(self.ell)))
            sample_cov_cmb = self.knox_covariance(dlth_cmb)
        if self.params['simu']['dust']:
            dlth_dust = Foreground(self.ell, self.nus).model_dust(Ad, alphad, betad, deltad, nu0_d)
            model += dlth_dust
            sample_cov_dust = np.zeros((self.nfreq, self.nfreq, len(self.ell), len(self.ell)))
            for i in range(self.nfreq):
                for j in range(i, self.nfreq):
                    sample_cov_dust[i][j] = self.knox_covariance(dlth_dust[i][j])
        if self.params['simu']['sync']:
            dlth_sync = Foreground(self.ell, self.nus).model_sync(As, alphas, betas, nu0_s)
            model += dlth_sync
            sample_cov_sync = np.zeros((self.nfreq, self.nfreq, len(self.ell), len(self.ell)))
            for i in range(self.nfreq):
                for j in range(i, self.nfreq):
                    sample_cov_sync[i][j] = self.knox_covariance(dlth_sync[i][j])
        if self.params['simu']['corr_dust_sync']:
            dlth_corr = Foreground(self.ell, self.nus).model_corr_dust_sync(Ad, alphad, betad, nu0_d, As, alphas, betas, nu0_s, eps)
            model += dlth_corr
            sample_cov_corr = np.zeros((self.nfreq, self.nfreq, len(self.ell), len(self.ell)))
            for i in range(self.nfreq):
                for j in range(i, self.nfreq):
                    sample_cov_corr[i][j] = self.knox_covariance(dlth_corr[i][j])

        # Define the noise model
        if self.params['simu']['noise'] is not True:
            noise_matrix = np.identity(np.shape(self.noise_cov_matrix))
        if self.params['fitting']['noise_covariance_matrix'] is not True:
            noise_matrix = np.zeros(np.shape(self.noise_cov_matrix)) 
            for i in range(self.nfreq):
                for j in range(i, self.nfreq):
                    for k in range(len(self.ell)):
                        noise_matrix[i][j][k][k] = (self.error_ps_noise[i][j][k])**2
        else :
            noise_matrix = self.noise_cov_matrix

        if self.params['fitting']['cmb_sample_variance']:
            noise_matrix += sample_cov_cmb
        if self.params['fitting']['dust_sample_variance']:
            noise_matrix += sample_cov_dust
        
        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                self.inv_noise_matrix = np.linalg.pinv(noise_matrix[i][j])
                loglike += - 0.5 * (self.mean_ps_sky[i][j] - model[i][j]).T @ self.inv_noise_matrix @ (self.mean_ps_sky[i][j] - model[i][j])
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
        
        print(self.simu_parameters)
        
        # Start the MCMC
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_prob_fn = self.loglikelihood, pool = pool, moves = [(emcee.moves.StretchMove(), self.params['MCMC']['stretch_move_factor']), (emcee.moves.DESnookerMove(gammas=self.params['MCMC']['snooker_move_gamma']), 1 - self.params['MCMC']['stretch_move_factor'])])
            sampler.run_mcmc(p0, mcmc_steps, progress=True)

        samples_flat = sampler.get_chain(flat = True, discard = self.params['MCMC']['discard'], thin = self.params['MCMC']['thin'])
        samples = sampler.get_chain()

        self.save_data(self.path_fit + '/fit_dict.pkl', {'nus':self.nus,
                              'ell':self.ell,
                              'samples': samples,
                              'samples_flat': samples_flat,
                              'fitted_parameters_names':self.sky_parameters_fitted_names,
                              'parameters': self.params,
                              'Dls' : self.power_spectra_sky,
                              'Nls' : self.power_spectra_noise,
                              'simulation_parameters': self.simu_parameters})        
        print("Fitting done !!!")














