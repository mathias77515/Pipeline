import numpy as np
import healpy as hp
from getdist import plots, MCSamples
import getdist
import yaml
import matplotlib.pyplot as plt
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator



class Plots:


    """
    
    Instance for plotting results of Monte-Carlo Markov Chain (i.e emcee). 

    """

    def __init__(self):

        pass

    def _make_samples(self, chain, names, labels):
        self.sample = MCSamples(samples=chain, names=names, labels=labels, ranges={'r':(0, None)})

    def make_list_free_parameter(self):

        '''
        
        Make few list :
            - fp       : list of value of free parameters
            - fp_name  : list of name for each values
            - fp_latex : list of name in LateX for each values

        '''

        fp = []
        fp_name = []
        fp_latex = []
        k = 0

        for iname, name in enumerate(self.params.keys()):
            for jname, n in enumerate(self.params[name]):
                if type(self.params[name][n]) is list:
                    if self.params[name][n][1] == 'f':
                        fp += [self.params[name][n][0]]
                        fp_latex += [self.params[name][n][2]]
                        fp_name += [list(self.params[name].keys())[k]]
                k += 1
            k = 0

        return fp, fp_name, fp_latex

    def _set_marker(self, values):
        
        '''
        
        Define the markers to see the input values in GetDist plot.
        
        '''
        
        dict = {}
        for ii, i in enumerate(values):
            #print(self.names[ii], values[ii])
            dict[self.names[ii]] = values[ii]

        if self.params['Sampler']['markers'] is False:
            dict = None
        return dict
    
    def get_convergence(self, chain):

        '''
        
        chain assumed to be not flat with shape (nsamples, nwalkers, nparams)
        
        '''

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.values, self.names, self.labels = self.make_list_free_parameter()

        plt.figure(figsize=(4, 4))

        for i in range(chain.shape[2]):
            plt.subplot(chain.shape[2], 1, i+1)
            plt.plot(chain[:, :, i], '-b', alpha=0.2)
            plt.plot(np.mean(chain[:, :, i], axis=1), '-r', alpha=1)
            plt.axhline(self.values[i], ls='--', color='black')
            plt.ylabel(self.names[i], fontsize=12)

        plt.xlabel('Iterations', fontsize=12)
        plt.show()

    def get_triangle(self, chain, names, labels):
        
        '''
        
        Make triangle plot of each estimated parameters
        
        '''

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.values, self.names, self.labels = self.make_list_free_parameter()
        self.marker = self._set_marker(self.values)
        print(self.marker)
        self._make_samples(chain, names, labels)

        plt.figure(figsize=(8, 8))
        # Triangle plot
        g = plots.get_subplot_plotter()
        g.triangle_plot([self.sample], filled=True, markers=self.marker, title_limit=self.params['Sampler']['title_limit'])
                #title_limit=1)
        plt.show()

class PlotsMM:

    def __init__(self, params):

        self.params = params
        self.stk = ['I', 'Q', 'U']

    def plot_FMM(self, m_in, m_out, center, seenpix, nus, figsize=(10, 8), istk=1, nsig=1, fwhm=0):
        
        C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
        plt.figure(figsize=figsize)

        k=1
        for i in range(self.params['QUBIC']['nrec']):
            hp.gnomview(C(m_in[i, :, istk]), rot=center, reso=15, cmap='jet', min = - nsig * np.std(m_out[0, seenpix, istk]), max = nsig * np.std(m_out[0, seenpix, istk]), sub=(self.params['QUBIC']['nrec'], 3, k),
                        title=r'Input - $\nu$ = '+f'{nus[i]:.0f} GHz')
            hp.gnomview(C(m_out[i, :, istk]), rot=center, reso=15, cmap='jet', min = - nsig * np.std(m_out[0, seenpix, istk]), max = nsig * np.std(m_out[0, seenpix, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+1),
                        title=r'Output - $\nu$ = '+f'{nus[i]:.0f} GHz')
            hp.gnomview(C(m_in[i, :, istk]) - C(m_out[i, :, istk]), rot=center, reso=15, cmap='jet', min = - nsig * np.std(m_out[0, seenpix, istk]), max = nsig * np.std(m_out[0, seenpix, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+2))

            k+=3
        plt.savefig(f'frequency_maps_{self.stk[istk]}.png')
        plt.close()