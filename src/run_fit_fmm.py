import numpy as np
import emcee
import yaml
import os
from schwimmbad import MPIPool
from multiprocessing import Pool
from pyoperators import MPI
import matplotlib.pyplot as plt

__path__ = os.path.dirname(os.path.abspath(__file__))

from lib.Qfoldertools import MergeAllFiles
from lib.Qspectra_component import SkySpectra

comm = MPI.COMM_WORLD

### Concatenate all realizations
files = MergeAllFiles('/Users/mregnier/Desktop/git/Pipeline/src/FMM/CMB_nrec2_new_code/spectrum/')

ell = files._reads_one_file(0, 'ell')
nus = files._reads_one_file(0, 'nus')[:2]

BBsignal = np.mean(files._reads_all_files('Dls'), axis=0)[:2, :2, :]
BBnoise = files._reads_all_files('Nl')[:, :2, :2, :]
BBsignal -= np.mean(BBnoise, axis=0)

class FitEllSpace:
    
    def __init__(self, x, y, yerr, model):
        
        self.x = x
        self.y = y
        self.yerr = yerr
        self.model = model
        
        with open(__path__ + '/FMM/fitting/fit_params.yaml', "r") as stream:
            self.params = yaml.safe_load(stream)
            
        ### Check if the user is giving the right dimensions
        self._check_shapein()
        
        ### Keep only the upper part of the matrix
        self._reshape_spectra()
        
        ### Reshape data to have two dimensions maximum
        self._init_reshape_data()
        
        ### Compute the noise covariance matrix
        self._get_noise_covariance_matrix()
    
    def _reshape_spectra_model(self, data):
        
        data_reshaped = np.zeros((self.nspecs, self.nbins))
        
        k=0
        for imap in range(self.nmaps):
            for jmap in range(imap, self.nmaps):
                data_reshaped[k, :] = data[imap, jmap]
                k+=1   
        
        return data_reshaped
    def _reshape_spectra(self):
        
        self.y_reshaped = np.zeros((self.nspecs, self.nbins))
        self.yerr_reshaped = np.zeros((self.nreals, self.nspecs, self.nbins))
        
        k=0
        for imap in range(self.nmaps):
            for jmap in range(imap, self.nmaps):
                self.y_reshaped[k, :] = self.y[imap, jmap]
                self.yerr_reshaped[:, k, :] = self.yerr[:, imap, jmap, :]
                k+=1
    def _check_shapein(self):

        if self.x.ndim != 1:
            raise TypeError('x should have 1 dimensions (Nbins)')
        
        if self.y.ndim != 3:
            raise TypeError('y should have 3 dimensions (Nmaps, Nmaps, Nbins)')
        
        if self.yerr.ndim != 4:
            raise TypeError('yerr should have 4 dimensions (Nreals, Nmaps, Nmaps, Nbins)')
        
        ### Get the shape (Nreals, Nmaps, Nmaps, Nbins)
        self.nreals, self.nmaps, _, self.nbins = self.yerr.shape
        self.nspecs = self.nmaps * (self.nmaps + 1) // 2
    def _init_reshape_data(self):
        
        
        
        self.y_reshaped = self.y_reshaped.reshape((self.nspecs*self.nbins))
        self.yerr_reshaped = self.yerr_reshaped.reshape((self.nreals, self.nspecs*self.nbins))
        
        print(self.y.shape, ' -> ', self.y_reshaped.shape)
        print(self.yerr.shape, ' -> ', self.yerr_reshaped.shape)
    def _get_noise_covariance_matrix(self):
        
        self.noise_covariance_matrix = np.cov(self.yerr_reshaped, rowvar=False)
        self.noise_correlation_matrix = np.corrcoef(self.yerr_reshaped, rowvar=False)
        
        self.noise_covariance_matrix += self._fill_sample_variance(self.y)
        
        self.invN = np.linalg.pinv(self.noise_covariance_matrix)   
    def _fill_sample_variance(self, bandpower):
        
        indices_tr = np.triu_indices(self.nmaps)
        matrix = np.zeros((self.nspecs, len(self.x), self.nspecs, len(self.x)))
        factor_modecount = 1/((2 * self.x + 1) * 0.015 * 30)
        
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                covar = (bandpower[i1, j1, :] * bandpower[i2, j2, :] + bandpower[i1, j2, :] * bandpower[i2, j1, :]) * factor_modecount
                matrix[ii, :, jj, :] = np.diag(covar)
        return matrix.reshape((self.nspecs*len(self.x), self.nspecs*len(self.x)))
    def _initial_conditions(self, nwalkers):
        
        x0 = np.zeros((0, nwalkers))
        keys = self.params.keys()
        
        for key in keys:
            params = self.params[key].keys()
            for param in params:
                
                ### Check if the user define the parameter as free and not fixed at given value
                if self.params[key][param]['fit'] is True and type(self.params[key][param]['fit']) == bool:
                    x0 = np.concatenate((x0, np.random.normal(self.params[key][param]['init_average'], self.params[key][param]['init_std'], (1, nwalkers))), axis=0)
                    
        self.ndim = x0.shape[0]
        
        return x0.T
    def log_prior(self):
        
        return 0
    def _fill_params(self, x):
        
        keys = self.params.keys()
        for key in keys:
            params = self.params[key].keys()
            for param in params:
                
                ### Add values of the fixed parameters only if there is no True in the params.yaml file
                if self.params[key][param]['fit'] is True:
                    pass
                else:
                    x = np.append(x, self.params[key][param]['fit'])

        return x
    def loglikelihood(self, x):
        
        x = self._fill_params(x)

        lp = self.log_prior()
        residuals = self.y_reshaped - self._reshape_spectra_model(self.model(*x)).reshape(self.y_reshaped.shape)
        #residuals = residuals.reshape(self.y_reshaped.shape)
        
        return lp - 0.5 * ((residuals).T @ self.invN @ (residuals)) 
    def run(self, nsteps, nwalkers, discard=0, comm=None):
        
        ### Initial condition for each parameters
        x0 = self._initial_conditions(nwalkers)

        with MPIPool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, log_prob_fn = self.loglikelihood, pool=pool)
            sampler.run_mcmc(x0, nsteps, progress=True)
        
        samples_flat = sampler.get_chain(flat = True, discard = discard, thin = 15)
        samples = sampler.get_chain()
        
        return samples, samples_flat

sky = SkySpectra(ell, nus)

fit = FitEllSpace(ell, BBsignal, BBnoise, model=sky.model)

samples, samples_flat = fit.run(200, 50, discard=100, comm=comm)

plt.figure()
plt.plot(samples[..., 0], '-k', alpha=0.1)
plt.axhline(0)
plt.show()

print()
print(f'Average : {np.mean(samples_flat, axis=0)}')
print(f'Error : {np.std(samples_flat, axis=0)}')
print()
