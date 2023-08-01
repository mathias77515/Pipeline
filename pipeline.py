import numpy as np
import yaml
import pickle
import time
import healpy as hp

from model.models import *
from likelihood.likelihood import *
from plots.plotter import *
import mapmaking.systematics as acq
from mapmaking.planck_timeline import *
from mapmaking.noise_timeline import *
import qubic
#from qubic import NamasterLib as nam
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from pyoperators import pcg, MPI


def save_pkl(name, d):
    with open(name, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)



class PipelineCrossSpectrum(Sky, Sampler, Plots):

    def __init__(self, nus, ell):

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.nus = nus
        self.ell = ell
        
        Sky.__init__(self, self.params, self.nus, self.ell)
        Sampler.__init__(self, self.params)
        Plots.__init__(self)

    def save_chain(self):

        """
        
        Method to save resulting chains coming from MCMC in pickle file.
        
        """

        save_pkl('output_chains.pkl', {'chain':self.chain, 
                                       'chain_flat':self.flat_chain,
                                       'true_values':self.value_free_params, 
                                       'true_names':self.name_free_params, 
                                       'true_names_latex':self.name_latex_free_params})
    def save_data(self, data, errors):

        """
        
        Save data in pickle files.
        
        """

        dict = {}
        k = 0
        for inu, nui in enumerate(self.nus):
            for jnu, nuj in enumerate(self.nus):
                print(f'Saving {nui:.0f}x{nuj:.0f} spectrum')
                dict[f'{nui:.0f}x{nuj:.0f}'] = data[k]
                dict[f'{nui:.0f}x{nuj:.0f}_errors'] = errors[k]
                k+=1
        
        save_pkl('data.pkl', dict)
    def open_pkl(self):
        with open(self.params['Data']['path'], 'rb') as handle:
            self.data = pickle.load(handle)
    def chi2(self, x):

        params = self.update_params(x)
        model_i = Sky(params, self.nus, self.ell)
        Dl_model_i = model_i.get_Dl()
        return - np.sum(((self.Dl_obs - Dl_model_i)/self.Dl_obs_error)**2)
    def _get_Dl_model(self, ell, x):
        
        params_true = self.update_params(x)
        model_i = Sky(params_true, self.nus, ell)
        return model_i.get_Dl()
    def gather_data(self):

        self.Dl_obs = np.zeros((len(self.nus)**2, len(self.ell)))
        self.Dl_obs_errors = np.zeros((len(self.nus)**2, len(self.ell)))
        Dl_obs = np.zeros((len(self.nus)**2, len(self.ell)))

        # Utilisation d'une compréhension de liste pour rassembler les valeurs en deux tableaux
        self.Dl_obs_error = np.vstack([self.data[key] for key in self.data.keys() if key.endswith('_errors')])
        self.Dl_obs = np.vstack([self.data[key] for key in self.data.keys() if not key.endswith('_errors')])
    def run(self):

        ### Loading data with errors
        self.open_pkl()
        self.gather_data()

        ### Define free parameters
        self.value_free_params, self.name_free_params, self.name_latex_free_params = self.make_list_free_parameter()
        
        ### MCMC
        self.chain, self.flat_chain = self.mcmc(self.value_free_params, self.chi2)
        self.fitted_params, self.fitted_params_errors = np.mean(self.flat_chain, axis=0), np.std(self.flat_chain, axis=0)
        
        ### Save result chains
        self.save_chain()

        ### Plots
        self.get_triangle(self.flat_chain, self.name_free_params, self.name_latex_free_params)
        self.get_convergence(self.chain)


class PipelineExternalData:

    def __init__(self, nside, skyconfig):
        
        self.external = self._read_external_nus()
        self.nside = nside
        self.skyconfig = skyconfig

    def _read_external_nus(self):

        allnus = [30, 44, 70, 100, 143, 217, 353]
        nus = []
        for i, name in enumerate(self.params['Data']['planck'].keys()):
            if self.params['Data']['planck'][name]:
                nus += [allnus[i]]
        return nus
    def read_pkl(self, name):
        
        with open(name, 'rb') as f:
            data = pickle.load(f)
        return data
    def _get_cmb(self, seed, r=0, Alens=1):
        
        mycls = acq.give_cl_cmb(r=r, Alens=Alens)

        np.random.seed(seed)
        cmb = hp.synfast(mycls, self.nside, verbose=False, new=True).T
        return cmb
    def _get_ave_map(self, central_nu, bw, nb=10):

        is_cmb = False
        model = []
        for key in self.skyconfig.keys():
            if key == 'cmb':
                is_cmb = True
            else:
                model += [self.skyconfig[key]]

        
        mysky = np.zeros((12*self.params['Sky']['nside']**2, 3))

        if len(model) != 0:
            sky = pysm3.Sky(nside=self.nside, preset_strings=model, output_unit="uK_CMB")
            edges_min = central_nu - bw/2
            edges_max = central_nu + bw/2
            bandpass_frequencies = np.linspace(edges_min, edges_max, nb) * u.GHz
            mysky += np.array(sky.get_emission(bandpass_frequencies)).T.copy()

        if is_cmb:
            cmb = self._get_cmb(self.skyconfig['cmb'])
            mysky += cmb.copy()
            
        return mysky
        
    def _get_fwhm(self, nu):
        return self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'fwhm{nu:.0f}']
    def _get_noise(self, nu):

        sigma = np.array([hp.ud_grade(self.read_pkl(f'data/Planck{nu:.0f}GHz.pkl')[f'noise{nu:.0f}'][:, i], self.params['Sky']['nside']) for i in range(3)]).T
        out = np.random.standard_normal(np.ones((12*self.params['Sky']['nside']**2, 3)).shape) * sigma
        return out
    def save_pkl(self, name, d):
        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run_external(self, fwhm=False, noise=True):

        self.maps = np.zeros((len(self.external), 12*self.nside**2, 3))

        for inu, nu in enumerate(self.external):
            self.maps[inu] = self._get_ave_map(nu, 10)
            if noise:
                self.maps[inu] += self._get_noise(nu)
            if fwhm:
                C = HealpixConvolutionGaussianOperator(fwhm=acq.arcmin2rad(self._get_fwhm(nu)))
                self.maps[inu]= C(self.maps[inu])

        with open(self.params['Data']['datafilename'], 'rb') as f:
            data = pickle.load(f)
        self.maps = np.concatenate((data['maps'], self.maps), axis=0)
        self.nus = np.concatenate((data['nus'], self.external), axis=0)

        self.save_pkl(self.params['Data']['datafilename'], {'maps':self.maps, 'nus':self.external})

class PipelineFrequencyMapMaking(ExternalData2Timeline):

    def __init__(self, comm):

        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)

        self.center = qubic.equ2gal(self.params['QUBIC']['RA_center'], self.params['QUBIC']['DEC_center'])
        self.fsub = int(self.params['QUBIC']['nsub'] / self.params['QUBIC']['nrec'])

        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        self.dict, self.dict_mono = self.get_dict()
        self.skyconfig = self._get_sky_config()
    
        ### Joint acquisition
        self.joint = acq.JointAcquisitionFrequencyMapMaking(self.dict, self.params['QUBIC']['type'], self.params['QUBIC']['nrec'], self.params['QUBIC']['nsub'])
        self.planck_acquisition143 = acq.PlanckAcquisition(143, self.joint.qubic.scene)
        self.planck_acquisition217 = acq.PlanckAcquisition(217, self.joint.qubic.scene)
        self.nus_eff = self._get_averaged_nus()

        ### Joint acquisition for TOD making
        self.joint_tod = acq.JointAcquisitionFrequencyMapMaking(self.dict, self.params['QUBIC']['type'], self.params['QUBIC']['nsub'], self.params['QUBIC']['nsub'])

        ExternalData2Timeline.__init__(self,
                                       self.skyconfig, 
                                       self.joint.qubic.allnus, 
                                       self.params['QUBIC']['nrec'], 
                                       nside=self.params['Sky']['nside'], 
                                       corrected_bandpass=self.params['QUBIC']['bandpass_correction'])
        
    def _get_averaged_nus(self):
        nus_eff = []

        for i in range(self.params['QUBIC']['nrec']):
            nus_eff += [np.mean(self.joint.qubic.allnus[i*self.fsub:(i+1)*self.fsub])]
        
        return np.array(nus_eff)
    def _get_sky_config(self):
        
        sky = {}
        for ii, i in enumerate(self.params['Sky'].keys()):
            #print(ii, i)

            if i == 'CMB':
                if self.params['Sky']['CMB']['cmb']:
                    sky['cmb'] = self.params['QUBIC']['seed']
            else:
                for jj, j in enumerate(self.params['Sky']['Foregrounds']):
                    #print(j, self.params['Foregrounds'][j])
                    if j == 'Dust':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['dust'] = self.params['QUBIC']['dust_model']
                    elif j == 'Synchrotron':
                        if self.params['Sky']['Foregrounds'][j]:
                            sky['synchrotron'] = self.params['QUBIC']['sync_model']

        return sky
    def get_ultrawideband_config(self):
    
        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
    
        return nu_ave, 2*delta/nu_ave
    def get_dict(self):
    
        '''
        Function for modify the qubic dictionary.
        '''

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()

        args = {'npointings':self.params['QUBIC']['npointings'], 
                'nf_recon':self.params['QUBIC']['nrec'], 
                'nf_sub':self.params['QUBIC']['nsub'], 
                'nside':self.params['Sky']['nside'], 
                'MultiBand':True, 
                'period':1, 
                'RA_center':self.params['QUBIC']['RA_center'], 
                'DEC_center':self.params['QUBIC']['DEC_center'],
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'comm':self.comm, 
                'nprocs_sampling':1, 
                'nprocs_instrument':self.size,
                'photon_noise':True, 
                'nhwp_angles':self.params['QUBIC']['nhwp_angles'], 
                'effective_duration':3, 
                'filter_relative_bandwidth':delta_nu_over_nu, 
                'type_instrument':'wide', 
                'TemperatureAtmosphere150':None, 
                'TemperatureAtmosphere220':None,
                'EmissivityAtmosphere150':None, 
                'EmissivityAtmosphere220':None, 
                'detector_nep':float(self.params['QUBIC']['detector_nep']), 
                'synthbeam_kmax':self.params['QUBIC']['synthbeam_kmax']}
        
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
    def _get_convolution(self):

        ### Define FWHMs
        if self.params['QUBIC']['convolution']:
            allfwhm = self.joint.qubic.allfwhm
            targets = np.array([])
            for irec in range(self.params['QUBIC']['nrec']):
                targets = np.append(targets, np.sqrt(allfwhm[irec*self.fsub:(irec+1)*self.fsub]**2 - np.min(allfwhm[irec*self.fsub:(irec+1)*self.fsub])**2))
                #targets = np.sqrt(allfwhm**2 - np.min(allfwhm)**2)
        else:
            targets = None
            allfwhm = None

        return targets, allfwhm
    def _get_tod(self):

        if self.params['QUBIC']['type'] == 'wide':
            if self.params['QUBIC']['nrec'] != 1:
                TOD_PLANCK = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
                for irec in range(int(self.params['QUBIC']['nrec']/2)):
                    if self.params['QUBIC']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                    TOD_PLANCK[irec] = C(self.maps[irec] + self.noise143)

                for irec in range(int(self.params['QUBIC']['nrec']/2), self.params['QUBIC']['nrec']):
                    if self.params['QUBIC']['convolution']:
                        C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*self.fsub:(irec+1)*self.fsub]))
                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)
        
                    TOD_PLANCK[irec] = C(self.maps[irec] + self.noise217)
            else:
                TOD_PLANCK = np.zeros((2*self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.allfwhm[-1])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)

                TOD_PLANCK[0] = C(self.maps[0] + self.noise143)
                TOD_PLANCK[1] = C(self.maps[0] + self.noise217)

            TOD_PLANCK = TOD_PLANCK.ravel()
            TOD_QUBIC = self.Hqtod(self.m_nu).ravel() + self.noiseq
            TOD = np.r_[TOD_QUBIC, TOD_PLANCK]

        else:

            sh_q = self.joint.qubic.ndets * self.joint.qubic.nsamples
            TOD_QUBIC = self.Hqtod(self.m_nu).ravel() + self.noiseq

            TOD_QUBIC150 = TOD_QUBIC[:sh_q].copy()
            TOD_QUBIC220 = TOD_QUBIC[sh_q:].copy()

            TOD = TOD_QUBIC150.copy()
    
            TOD_PLANCK = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))
            for irec in range(int(self.params['QUBIC']['nrec']/2)):
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=np.min(allfwhm[irec*f:(irec+1)*f]))
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)
        
            TOD = np.r_[TOD, C(self.maps[irec] + self.noise143).ravel()]

            TOD = np.r_[TOD, TOD_QUBIC220.copy()]
            for irec in range(int(self.params['QUBIC']['nrec']/2), self.params['QUBIC']['nrec']):
                if self.params['QUBIC']['convolution']:
                    C = HealpixConvolutionGaussianOperator(fwhm=np.min(self.allfwhm[irec*f:(irec+1)*f]))
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)
        
            TOD = np.r_[TOD, C(self.maps[irec] + self.noise217).ravel()]

        self.m_nu_in = np.zeros((self.params['QUBIC']['nrec'], 12*self.params['Sky']['nside']**2, 3))

        for i in range(self.params['QUBIC']['nrec']):
            self.m_nu_in[i] = np.mean(self.m_nu[i*self.fsub:(i+1)*self.fsub], axis=0)

        return TOD
    def _barrier(self):

        if self.comm is None:
            pass
        else:
            self.comm.Barrier()
    def print_message(self, message):
        if self.comm is None:
            print(message)
        else:
            if self.rank == 0:
                print(message)
    def _pcg(self, d):

        '''
        
        Solve the map-making equation iteratively :     H^T . N^{-1} . H . x = H^T . N^{-1} . d

        The PCG used for the minimization is intrinsequely parallelized (e.g see PyOperators).
        
        '''


        A = self.H.T * self.invN * self.H
        b = self.H.T * self.invN * d

        ### Preconditionning
        M = acq.get_preconditioner(np.ones(12*self.params['Sky']['nside']**2))

        ### PCG
        start = time.time()
        solution_qubic_planck = pcg(A, b, x0=None, M=M, tol=self.params['PCG']['tol'], disp=True, maxiter=self.params['PCG']['maxiter'])

        if self.params['QUBIC']['nrec'] == 1:
            solution_qubic_planck['x'] = np.array([solution_qubic_planck['x']])
        end = time.time()
        execution_time = end - start
        self.print_message(f'Simulation done in {execution_time:.3f} s')

        return solution_qubic_planck['x']
    def save_data(self, name, d):

        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run_MM(self):

        ### Coverage map
        self.coverage = self.joint.qubic.subacqs[0].get_coverage()
        covnorm = self.coverage / self.coverage.max()
        self.seenpix = covnorm > self.params['QUBIC']['covcut']
        self.seenpix_for_plot = covnorm > 0
        self.mask = np.ones(12*self.params['Sky']['nside']**2)
        self.mask[self.seenpix] = self.params['QUBIC']['kappa']
        self.mask_nam = self.mask.copy()
        self.mask_nam[~self.seenpix] = self.params['QUBIC']['kappa']
        #self.Namaster = nam.Namaster(self.mask_nam,
        #                                           lmin=self.params['Spectrum']['lmin'],
        #                                           lmax=self.params['Sky']['nside']*2,
        #                                           delta_ell=self.params['Spectrum']['dl'],
        #                                           aposize=self.params['Spectrum']['aposize'])
        #print(self.Namaster)
        #stop
        
        ### Angular resolutions
        self.targets, self.allfwhm = self._get_convolution()
        
        ### Define reconstructed and TOD operator
        self.H = self.joint.get_operator(fwhm=self.targets)
        self.Htod = self.joint_tod.get_operator(fwhm=self.allfwhm)
        self.Hqtod = self.joint_tod.qubic.get_operator(fwhm=self.allfwhm)
        
        ### Inverse noise covariance matrix
        self.invN = self.joint.get_invntt_operator(mask=self.mask)

        ### Noises
        seed_noise_planck = 42
        self.noise143 = self.planck_acquisition143.get_noise(seed_noise_planck) * self.params['Data']['level_planck_noise']
        self.noise217 = self.planck_acquisition217.get_noise(seed_noise_planck) * self.params['Data']['level_planck_noise']

        if self.params['QUBIC']['type'] == 'two':
            qubic_noise = QubicDualBandNoise(self.dict, self.params['QUBIC']['npointings'], self.params['QUBIC']['detector_nep'])
        elif self.params['QUBIC']['type'] == 'wide':
            qubic_noise = QubicWideBandNoise(self.dict, self.params['QUBIC']['npointings'], self.params['QUBIC']['detector_nep'])

        self.noiseq = qubic_noise.total_noise(self.params['QUBIC']['ndet'], 
                                       self.params['QUBIC']['npho150'], 
                                       self.params['QUBIC']['npho220']).ravel()
        
        
        ### Get simulated data
        TOD = self._get_tod()

        ### Wait for all processes
        self._barrier()

        ### Solve map-making equation
        self.s_hat = self._pcg(TOD)

        ### Save maps
        self.save_data(self.params['Data']['datafilename'], {'maps':self.s_hat, 'nus':self.nus_eff})

        
class PipelineEnd2End(PipelineFrequencyMapMaking, PipelineExternalData):

    def __init__(self):

        comm = MPI.COMM_WORLD

    
        PipelineFrequencyMapMaking.__init__(self, comm)
        PipelineExternalData.__init__(self, self.params['Sky']['nside'], self.skyconfig)

        self._break_pipeline()

    def _update_data(self, dict, key, value):

        dict[key] = value

        self.save_pkl(self.params['Data']['datafilename'], dict)

    def _break_pipeline(self):

        if self.params['Method'] != 'FMM' and self.params['Method'] != 'CMM':
            raise TypeError('Choose FMM or CMM.')
        
    def main(self):
        
        if self.params['Method'] == 'FMM':
            
            ### Execute Frequency Map-Making
            self.run_MM()
        
            if self.params['Data']['use_external_data']:
                ### Create Fake frequency maps from external data
                self.run_external(fwhm=self.params['QUBIC']['convolution'], noise=True)

            PlotsMM(self.params).plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix_for_plot, self.nus_eff, istk=0, fwhm=0.0078)
            PlotsMM(self.params).plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix_for_plot, self.nus_eff, istk=1, fwhm=0.0078)
            PlotsMM(self.params).plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix_for_plot, self.nus_eff, istk=2, fwhm=0.0078)

        ### Read maps
        #m_nu_q = self._read_maps_qubic()
        #m_nu_ext = self._read_maps_external()
        #print(m_nu_q['maps'].shape)
        #print(m_nu_q['nus'])
        #print(m_nu_ext['maps'].shape)
        #print(m_nu_q['nus'])
        #print(m_nu_ext['nus'])




        

        

        
        


