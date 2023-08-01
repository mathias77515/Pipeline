import numpy as np
import healpy as hp
import fgb.mixing_matrix as mm
import fgb.component_model as c
import matplotlib.pyplot as plt

def separate_dictionaries(input_dict):
    cmb_dict = input_dict.get('CMB', {})
    foregrounds_dict = input_dict.get('Foregrounds', {})
    return cmb_dict, foregrounds_dict

class CMBModel:

    """
    
    CMB description assuming parametrized emission law such as :

        Dl_CMB = r * Dl_tensor_r1 + Alens * Dl_lensed
        
        Parameters
        -----------
            - params : Dictionary coming from `params.yml` file that define every parameters
            - ell    : Multipole used during the analysis
    
    """

    def __init__(self, params, ell):
        
        self.params = params
        self.ell = ell
    
    def give_cl_cmb(self):
        power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if self.params['CMB']['Alens'][0] != 1.:
            power_spectrum *= self.params['CMB']['Alens'][0]
        if self.params['CMB']['r'][0]:
            power_spectrum += self.params['CMB']['r'][0] * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return power_spectrum

    def cl2dl(self, ell, cl):

        dl=np.zeros(ell.shape[0])
        for i in range(ell.shape[0]):
            dl[i]=(ell[i]*(ell[i]+1)*cl[i])/(2*np.pi)
        return dl
    
    def get_Dl_cmb(self):
        allDl = self.cl2dl(np.arange(1, 4001, 1), self.give_cl_cmb()[2])
        Dl_eff = np.interp(self.ell, np.arange(1, 4001, 1), allDl)
        return Dl_eff

class ForeGroundModels(CMBModel):

    """
    
    Foreground models assuming parametrized emission law such as :

        Dl_FG = Ad * Delta_d * fnu1d * fnu2d * (ell/80)**alpha_d +
                As * Delta_s * fnu1s * fnu2s * (ell/80)**alpha_s + 
                eps * sqrt(Ad * As) * Delta_d * (fnu1d * fnu2s + fnu2d * fnu1s) * (ell/80)**((alpha_d + alpha_s)/2)
        
        Parameters
        -----------
            - params : Dictionary coming from `params.yml` file that define every parameters
            - nus    : Array that contain every frequencies for the analysis
            - ell    : Multipole used during the analysis
    
    """

    def __init__(self, params, nus, ell):
        
        CMBModel.__init__(self, params, ell)
        
        self.nus = nus

    def dust_model(self, ell, fnu1, fnu2):
        return self.params['Foregrounds']['Ad'][0] * self.params['Foregrounds']['deltad'][0] * fnu1 * fnu2 * (ell/80)**self.params['Foregrounds']['alphad'][0]
    
    def sync_model(self, ell, fnu1, fnu2):
        return self.params['Foregrounds']['As'][0] * self.params['Foregrounds']['deltas'][0] * fnu1 * fnu2 * (ell/80)**self.params['Foregrounds']['alphas'][0]
    
    def dustsync_model(self, ell, fnu1d, fnu2d, fnu1s, fnu2s):
        m = self.params['Foregrounds']['eps'][0] * np.sqrt(abs(self.params['Foregrounds']['Ad'][0] * self.params['Foregrounds']['As'][0])) * \
            (fnu1d*fnu2s + fnu1s*fnu2d) * (ell/80)**((self.params['Foregrounds']['alphad'][0] + self.params['Foregrounds']['alphas'][0])/2)
        return m
    
    def scale_dust(self, nu, temp=20, beta_d=1.54):
    
        comp = c.Dust(nu0=self.params['Foregrounds']['nu0_d'], temp=temp, beta_d=beta_d)
    
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]
        return A
    
    def scale_sync(self, nu, beta_pl=-3):
    
        comp = c.Synchrotron(nu0=self.params['Foregrounds']['nu0_s'], beta_pl=beta_pl)
    
        A = mm.MixingMatrix(comp).evaluator(np.array([nu]))()[0]
        return A
    
    def get_Dl_fg(self, ell, fnu1d, fnu2d, fnu1s, fnu2s):
        


        m = np.zeros(len(ell))

        if self.params['Foregrounds']['Dust']:
            m += self.dust_model(ell, fnu1d, fnu2d)

        if self.params['Foregrounds']['Synchrotron']:
            m += self.sync_model(ell, fnu1s, fnu2s)

        if self.params['Foregrounds']['DustSync']:
            m += self.dustsync_model(ell, fnu1d, fnu2d, fnu1s, fnu2s)

        return m


class Sky(ForeGroundModels):

    """
    
    Sky description for CMB + Foregrouds model assuming parametrized emission law. 
        
        Parameters
        -----------
            - params : Dictionary coming from `params.yml` file that define every parameters
            - nus    : Array that contain every frequencies for the analysis
            - ell    : Multipole used during the analysis
    
    """

    def __init__(self, params, nus, ell):

        ForeGroundModels.__init__(self, params, nus, ell)

    def model(self, fnu1d, fnu2d, fnu1s, fnu2s):

        Dl_cmb = self.get_Dl_cmb()
        Dl_fg = self.get_Dl_fg(self.ell, fnu1d, fnu2d, fnu1s, fnu2s)

        return Dl_cmb + Dl_fg
    
    def make_list_free_parameter(self):
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
    
    def update_params(self, new_params):
        k = 0
        for iname, name in enumerate(self.params.keys()):
            for jname, n in enumerate(self.params[name]):
                if type(self.params[name][n]) is list:
                    if self.params[name][n][1] == 'f':

                        self.params[name][n][0] = new_params[k]
                        k+=1
        return self.params
        
    def get_Dl(self):

        fd = np.zeros(len(self.nus))
        fs = np.zeros(len(self.nus))
        for inu, nu in enumerate(self.nus):
            fd[inu] = self.scale_dust(nu)
            fs[inu] = self.scale_sync(nu)
        
        Dl = np.zeros((len(self.nus)*len(self.nus), len(self.ell)))

        k = 0
        for inu, nu in enumerate(self.nus):
            for jnu, nu in enumerate(self.nus):
                
                Dl[k] = self.model(fd[inu], fd[jnu], fs[inu], fs[jnu])

                k += 1

        return Dl
    
    def _plot_Dl(self, Dl, Dl_errors, model=None, model_fit=None, model_fit_err=None, figsize=(8, 8), fmt='or', title=None):
        

        num_dl, num_bin = Dl.shape
        num_nus = int(np.sqrt(num_dl))
        
        ell_min, ell_max = self.ell.min(), self.ell.max()
        ell = np.linspace(ell_min, ell_max, model.shape[1])

        plt.figure(figsize=figsize)

        k = 0
        for _ in range(num_nus):
            for _ in range(num_nus):
                plt.subplot(num_nus, num_nus, k+1)
                plt.errorbar(self.ell, Dl[k], yerr=Dl_errors[k], fmt=fmt, capsize=3)
                if model is not None:
                    plt.plot(ell, model[k], '-k', label='Model')
                if model_fit is not None:
                    plt.plot(ell, model_fit[k], '--b', label='Fit')
                    if model_fit_err is not None:
                        plt.fill_between(ell, model_fit[k] - model_fit_err[k]/2, model_fit[k] + model_fit_err[k]/2, color='blue', alpha=0.2)
                if title is not None:
                    plt.title(title[k])

                k+=1

                if k == 1:
                    plt.legend(frameon=False, fontsize=12)
        plt.show()


