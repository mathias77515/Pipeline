import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

def _plot_reconstructed_maps(maps, m_in, name_file, center, reso=15, figsize=(12, 8), fwhm=0, min=-8, max=8):
    
    """
    
    Save a PNG with the actual maps at iteration i. It assumes that maps is 3-dimensional
    
    """
    stk = ['I', 'Q', 'U']
    plt.figure(figsize=figsize)
    
    _shape = maps.shape
    index = np.where(maps[0, :, 0] != hp.UNSEEN)[0]
    index_not_seen = np.where(maps[0, :, 0] == hp.UNSEEN)[0]
    C = HealpixConvolutionGaussianOperator(fwhm=fwhm, lmax=3*hp.npix2nside(m_in.shape[1]))
    res = maps - m_in
    res
    k=0
    for inu in range(_shape[0]):
        _m = C(maps[inu])
        _r = C(res[inu])
        _m[index_not_seen, :] = hp.UNSEEN
        _r[index_not_seen, :] = hp.UNSEEN
        for istk in range(_shape[-1]):

            hp.gnomview(_m[:, istk], rot=center, reso=reso, cmap='jet', sub=(_shape[0], _shape[-1]*2, k+1),
                        notext=True, min=min[istk], max=max[istk], title=f'Output - {stk[istk]}')
            hp.gnomview(_r[:, istk], rot=center, reso=reso, cmap='jet', sub=(_shape[0], _shape[-1]*2, k+2),
                        notext=True, min=min[istk], max=max[istk], title=f'Residuals - {stk[istk]}')
            k+=2
    
    plt.savefig(name_file)
    plt.close()

class Plots:


    """
    
    Instance for plotting results of Monte-Carlo Markov Chain (i.e emcee). 

    """

    def __init__(self):

        pass

    def _make_samples(self, chain, names, labels):
        self.sample = MCSamples(samples=chain, names=names, labels=labels)
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

        for iname, name in enumerate(self.params['Sky'].keys()):
            try:
                #print('yes')
                for jname, n in enumerate(self.params['Sky'][name]):
                    if type(self.params['Sky'][name][n]) is list:
                        #print(self.params['Sky'][name][n], n)
                        if self.params['Sky'][name][n][1] == 'f':
                            fp += [self.params['Sky'][name][n][0]]
                            fp_latex += [self.params['Sky'][name][n][2]]
                            fp_name += [list(self.params['Sky'][name].keys())[k]]
                    k += 1
                k = 0
            except: pass

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
    def get_convergence(self, chain, job_id):

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
        plt.savefig(f'allplots_{job_id}/Convergence_chain.png')
        plt.close()
    def get_triangle(self, chain, names, labels, job_id):
        
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
        plt.savefig(f'allplots_{job_id}/triangle_plot.png')
        plt.close()
    def get_Dl_plot(self, ell, Dl, Dl_err, nus, job_id, figsize=(10, 10), model=None):

        plt.figure(figsize=figsize)

        k=0
        for i in range(len(nus)):
            for j in range(len(nus)):
                plt.subplot(len(nus), len(nus), k+1)
                plt.errorbar(ell, Dl[k], yerr=Dl_err[k], fmt='or')
                if model is not None:
                    plt.errorbar(ell, model[k], fmt='-k')
                plt.title(f'{nus[i]:.0f}x{nus[j]:.0f}')
                #plt.yscale('log')
                k+=1

        plt.tight_layout()
        plt.savefig(f'allplots_{job_id}/Dl_plot.png')
        plt.close()




class PlotsFMM:

    def __init__(self, seenpix):

        self.stk = ['I', 'Q', 'U']
        self.seenpix = seenpix

    def plot_frequency_maps(self, m_in, m_out, center, reso=15, nsig=3, filename=None, figsize=(10, 8)):
        
        m_in[:, ~self.seenpix, :] = hp.UNSEEN
        m_out[:, ~self.seenpix, :] = hp.UNSEEN
        res = m_out - m_in
        nf, _, _ = m_out.shape
        
        fig = plt.figure(figsize=figsize)
        
        k=1
        for inu in range(nf):
            for istk in range(3):

                sig = np.std(m_out[0, self.seenpix, istk])
                
                hp.gnomview(m_out[inu, :, istk], rot=center, reso=reso, cmap='jet', 
                        min = - nsig * sig, 
                        max = nsig * sig, 
                        sub=(nf, 6, k),
                        title=f'Output {self.stk[istk]}', notext=True)
                
                hp.gnomview(res[inu, :, istk], rot=center, reso=reso, cmap='jet', 
                        min = - nsig * sig, 
                        max = nsig * sig, 
                        sub=(nf, 6, k+1),
                        title=f'Residuals {self.stk[istk]}', notext=True)
                
                
                #plt.suptitle(r'$\nu$ = '+f'{nus[inu]:.1f} GHz', x=0.5, y=1 - inu/10)
                
                k+=2
            #plt.text(-2.3, 1 - inu/10, r'$\nu$ = '+f'{nus[inu]:.1f} GHz', fontsize=15,
            #        horizontalalignment='center')
            
        if filename is not None:
            plt.savefig(filename)
        plt.close()
        
    def plot_FMM_old(self, m_in, m_out, center, seenpix, nus, job_id, figsize=(10, 8), istk=1, nsig=3, name='signal'):
        
        m_in[:, ~seenpix, :] = hp.UNSEEN
        m_out[:, ~seenpix, :] = hp.UNSEEN

        
        plt.figure(figsize=figsize)

        k=1
        for i in range(self.params['QUBIC']['nrec']):
            
            hp.gnomview(m_in[i, :, istk], rot=center, reso=15, cmap='jet', 
                        min = - nsig * np.std(m_out[0, seenpix, istk]), 
                        max = nsig * np.std(m_out[0, seenpix, istk]), 
                        sub=(self.params['QUBIC']['nrec'], 3, k),
                        title=r'Input - $\nu$ = '+f'{nus[i]:.0f} GHz')
            hp.gnomview(m_out[i, :, istk], rot=center, reso=15, cmap='jet', 
                        min = - nsig * np.std(m_out[0, seenpix, istk]), 
                        max = nsig * np.std(m_out[0, seenpix, istk]), 
                        sub=(self.params['QUBIC']['nrec'], 3, k+1),
                        title=r'Output - $\nu$ = '+f'{nus[i]:.0f} GHz')
            
            res = m_in[i, :, istk] - m_out[i, :, istk]
            res[~seenpix] = hp.UNSEEN
            
            hp.gnomview(res, rot=center, reso=15, cmap='jet', min = - nsig * np.std(m_out[0, seenpix, istk]), max = nsig * np.std(m_out[0, seenpix, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+2))

            k+=3
        plt.savefig(f'FMM/allplots_{job_id}/frequency_maps_{self.stk[istk]}_{name}.png')
        plt.close()

    def plot_FMM_mollview(self, m_in, m_out, nus, job_id, figsize=(10, 8), istk=1, nsig=3, fwhm=0):

        C = HealpixConvolutionGaussianOperator(fwhm=fwhm)
        plt.figure(figsize=figsize)

        k=1
        for i in range(self.params['QUBIC']['nrec']):
            hp.mollview(C(m_in[i, :, istk]), cmap='jet', 
            min = - nsig * np.std(m_out[0, :, istk]), 
            max = nsig * np.std(m_out[0, :, istk]), sub=(self.params['QUBIC']['nrec'], 3, k),
                        title=r'Input - $\nu$ = '+f'{nus[i]:.0f} GHz')

            hp.mollview(C(m_out[i, :, istk]), cmap='jet', 
            min = - nsig * np.std(m_out[0, :, istk]), 
            max = nsig * np.std(m_out[0, :, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+1),
                        title=r'Output - $\nu$ = '+f'{nus[i]:.0f} GHz')

            hp.mollview(C(m_in[i, :, istk]) - C(m_out[i, :, istk]), cmap='jet', 
            min = - nsig * np.std(m_out[0, :, istk]), 
            max = nsig * np.std(m_out[0, :, istk]), sub=(self.params['QUBIC']['nrec'], 3, k+2))

            k+=3
        plt.savefig(f'allplots_{job_id}/frequency_maps_{self.stk[istk]}_moll.png')
        plt.close()