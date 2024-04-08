#### General packages
import numpy as np
import os
import yaml
import pickle
 
#### QUBIC packages
import qubic
from qubic import NamasterLib as nam
from qubic.beams import BeamGaussian
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator


class Spectrum:
    '''
    Class to compute the different spectra for our realisations
    '''

    def __init__(self, file, verbose=True):
        
        
        path, filename = os.path.split(file)
        self.jobid = filename.split('_')[1].split('.')[0]
        print(f'Job id found : ', self.jobid)
        with open('params.yml', "r") as stream:
            self.params = yaml.safe_load(stream)
        
        self.path_spectrum = os.path.join(os.path.dirname(os.path.dirname(file)), "spectrum")
        
        with open(file, 'rb') as f:
            self.dict_file = pickle.load(f)
        #print(self.dict_file)
        #stop
        if not os.path.isdir(self.path_spectrum):
            os.makedirs(self.path_spectrum)
        
        
        self.verbose = verbose
        self.sky_maps = self.dict_file['maps']
        self.noise_maps = self.dict_file['maps_noise']
        self.nus = self.dict_file['nus']
        self.nfreq = len(self.nus)
        self.nrec = self.params['QUBIC']['nrec']
        self.fsub = self.params['QUBIC']['fsub']
        self.nside = self.params['Sky']['nside']
        self.nsub = int(self.fsub * self.nrec)
        self.my_dict, _ = self.get_dict()

        # Define Namaster class
        coverage = self.dict_file['coverage']
        seenpix = coverage/np.max(coverage) < 0.2
        self.namaster = nam.Namaster(weight_mask = list(~np.array(seenpix)),
                                 lmin = self.params['Spectrum']['lmin'],
                                 lmax = self.params['Spectrum']['lmax'],
                                 delta_ell = self.params['Spectrum']['dl'])

        self.ell = self.namaster.get_binning(self.params['Sky']['nside'])[0]
        
        if self.params['QUBIC']['convolution'] is True:
            fwhm = self.allfwhm()
            print(fwhm)
            allfwhm = np.zerso(self.nfreq)
            for i in range(self.nrec):
                allfwhm[i] = fwhm[(i+1)*self.fsub - 1]
            self.allfwhm = allfwhm
        else:
            self.allfwhm = np.zeros(self.nfreq)

    def get_dict(self):
        """
        Method to modify the qubic dictionary.
        """

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()

        args = {'npointings':self.params['QUBIC']['npointings'], 
                'nf_recon':self.nrec, 
                'nf_sub':self.nsub, 
                'nside':self.nside, 
                'MultiBand':True, 
                'period':1, 
                'RA_center':self.params['QUBIC']['RA_center'], 
                'DEC_center':self.params['QUBIC']['DEC_center'],
                'filter_nu':nu_ave*1e9, 
                'noiseless':False, 
                'dtheta':self.params['QUBIC']['dtheta'],
                'nprocs_sampling':1,
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
    def get_ultrawideband_config(self):
        """
        Method that pre-compute UWB configuration.
        """

        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave
        return nu_ave, 2*delta/nu_ave    
    def synthbeam(self, synthbeam_peak150_fwhm, dtype=np.float32):
        sb = SyntheticBeam()
        sb.dtype = np.dtype(dtype)
        nripples = self.my_dict['nripples']
        synthbeam_peak150_fwhm = np.radians(self.my_dict['synthbeam_peak150_fwhm'])
        if not nripples:
            sb.peak150 = BeamGaussian(synthbeam_peak150_fwhm)
        else:
            sb.peak150 = BeamGaussianRippled(synthbeam_peak150_fwhm,
                                             nripples=nripples)
        return sb    
    def allfwhm(self):
        '''
        Function to compute the fwhm for all sub bands.

        Return :
            - allfwhm (list [nrec * nsub])
        '''

        return 0.39268176 * (150 / self.nus)
    def compute_auto_spectrum(self, map, fwhm):
        '''
        Function to compute the auto-spectrum of a given map

        Argument : 
            - map(array) [nrec/ncomp, npix, nstokes] : map to compute the auto-spectrum
            - allfwhm(float) : in radian
        Return : 
            - (list) [len(ell)] : BB auto-spectrum
        '''

        DlBB = self.namaster.get_spectra(map=map.T, map2=None, verbose=False, beam_correction = np.rad2deg(fwhm))[1][:, 2]
        return DlBB
    def compute_cross_spectrum(self, map1, fwhm1, map2, fwhm2):
        '''
        Function to compute cross-spectrum, taking into account the different resolution of each sub-bands

        Arguments :
            - map1 & map2 (array [nrec/ncomp, npix, nstokes]) : the two maps needed to compute the cross spectrum
            - fwhm1 & fwhm2 (float) : the respective fwhm for map1 & map2 in radian

        Return : 
            - (list) [len(ell)] : BB cross-spectrum
        '''

        # Put the map with the highest resolution at the worst one before doing the cross spectrum
        # Important because the two maps had to be at the same resolution and you can't increase the resolution
        if fwhm1<fwhm2 :
            C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm2**2 - fwhm1**2))
            convoluted_map = C*map1
            return self.namaster.get_spectra(map=convoluted_map.T, map2=map2.T, verbose=False, beam_correction = np.rad2deg(fwhm2))[1][:, 2]
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(fwhm1**2 - fwhm2**2))
            convoluted_map = C*map2
            return self.namaster.get_spectra(map=map1.T, map2=convoluted_map.T, verbose=False, beam_correction = np.rad2deg(fwhm1))[1][:, 2]
    def compute_array_power_spectra(self, maps):
        ''' 
        Function to fill an array with all the power spectra computed

        Argument :
            - maps (array [nreal, nrec/ncomp, npix, nstokes]) : all your realisation maps

        Return :
            - power_spectra_array (array [nrec/ncomp, nrec/ncomp]) : element [i, i] is the auto-spectrum for the reconstructed sub-bands i 
                                                                     element [i, j] is the cross-spectrum between the reconstructed sub-band i & j
        '''

        power_spectra_array = np.zeros((self.nfreq, self.nfreq, len(self.ell)))

        for i in range(self.nfreq):
            for j in range(i, self.nfreq):
                print(f'====== {self.nus[i]:.0f}x{self.nus[j]:.0f} ======')
                #print(self.allfwhm)
                if i==j :
                    # Compute the auto-spectrum
                    power_spectra_array[i,j] = self.compute_auto_spectrum(maps[i], self.allfwhm[i])
                else:
                    # Compute the cross-spectrum
                    power_spectra_array[i,j] = self.compute_cross_spectrum(maps[i], self.allfwhm[i], maps[j], self.allfwhm[j])
                #print(power_spectra_array[i, j])
        return power_spectra_array
    def compute_power_spectra(self):
        '''
        Function to compute the power spectra array for the sky and for the noise realisations

        Return :
            - sky power spectra array (array [nrec/ncomp, nrec/ncomp])
            - noise power spectra array (array [nrec/ncomp, nrec/ncomp])
        '''

        sky_power_spectra = self.compute_array_power_spectra(self.sky_maps)
        noise_power_spectra = self.compute_array_power_spectra(self.noise_maps)
        return sky_power_spectra, noise_power_spectra
    def save_data(self, name, d):

        """
        
        Method to save data using pickle convention.
        
        """
        
        with open(name, 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def run(self):
        
        self.Dl, self.Nl = self.compute_power_spectra()
        
        print('Power spectra computed !!!')
        
        self.save_data(self.path_spectrum + '/' + f'spectrum_{self.jobid}.pkl', {'nus':self.nus,
                              'ell':self.ell,
                              'Dls':self.Dl,
                              'Nl':self.Nl})
