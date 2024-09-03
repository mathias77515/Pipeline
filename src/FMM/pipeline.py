### General packages
import os
import pickle
import time

import numpy as np
import qubic
import yaml
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

### Local packages
from lib.Qacquisition import *
from lib.Qcg import *
from lib.Qcomponent_model import *
from lib.Qfoldertools import *
from lib.Qmap_plotter import *
from lib.Qnoise import *
from lib.Qspectra import *

from .model.externaldata import *
from .model.models import *
from .model.planck_timeline import *


def save_pkl(name, d):
    with open(name, "wb") as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


__all__ = ["PipelineFrequencyMapMaking", "PipelineEnd2End"]


class PipelineFrequencyMapMaking:
    """

    Instance to reconstruct frequency maps using QUBIC abilities.

    Parameters :
    ------------
        - comm : MPI communicator
        - file : str to create folder for data saving

    """

    def __init__(self, comm, file, params):

        self.mapmaking_time_0 = time.time()

        self.params = params

        ### Fsub
        self.fsub = int(self.params["QUBIC"]["nsub_out"] / self.params["QUBIC"]["nrec"])

        self.file = file
        self.plot_folder = "src/FMM/" + self.params["path_out"] + "png/"

        self.externaldata = PipelineExternalData(file, self.params)
        self.externaldata.run(fwhm=self.params["QUBIC"]["convolution_in"], noise=True)

        self.externaldata_noise = PipelineExternalData(
            file, self.params, noise_only=True
        )
        self.externaldata_noise.run(
            fwhm=self.params["QUBIC"]["convolution_in"], noise=True
        )

        if comm.Get_rank() == 0:
            if not os.path.isdir("src/FMM/" + self.params["path_out"] + "maps/"):
                os.makedirs("src/FMM/" + self.params["path_out"] + "maps/")
            if not os.path.isdir("src/FMM/" + self.params["path_out"] + "png/"):
                os.makedirs("src/FMM/" + self.params["path_out"] + "png/")

        self.job_id = os.environ.get("SLURM_JOB_ID")
        self.center = qubic.equ2gal(
            self.params["SKY"]["RA_center"], self.params["SKY"]["DEC_center"]
        )

        ### MPI common arguments
        self.comm = comm
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

        ### Sky
        self.dict_in, self.dict_mono_in = self.get_dict(key="in")
        self.dict_out, self.dict_mono_out = self.get_dict(key="out")
        self.skyconfig = self._get_sky_config()

        ### Joint acquisition for TOD making
        self.joint_tod = JointAcquisitionFrequencyMapMaking(
            self.dict_in,
            self.params["QUBIC"]["instrument"],
            self.params["QUBIC"]["nsub_in"],
            self.params["QUBIC"]["nsub_in"],
            H=None,
        )

        ### Joint acquisition
        if self.params["QUBIC"]["nsub_in"] == self.params["QUBIC"]["nsub_out"]:
            H = self.joint_tod.qubic.H
        else:
            H = None

        self.joint = JointAcquisitionFrequencyMapMaking(
            self.dict_out,
            self.params["QUBIC"]["instrument"],
            self.params["QUBIC"]["nrec"],
            self.params["QUBIC"]["nsub_out"],
            H=H,
        )

        self.planck_acquisition143 = PlanckAcquisition(143, self.joint.qubic.scene)
        self.planck_acquisition217 = PlanckAcquisition(217, self.joint.qubic.scene)
        self.nus_Q = self._get_averaged_nus()

        ### Coverage map
        self.coverage = self.joint.qubic.subacqs[0].get_coverage()
        covnorm = self.coverage / self.coverage.max()
        self.seenpix = covnorm > self.params["SKY"]["coverage_cut"]
        self.seenpix_qubic = covnorm > 0
        self.fsky = self.seenpix.astype(float).sum() / self.seenpix.size
        self.coverage_cut = self.coverage.copy()
        self.coverage_cut[~self.seenpix] = 1

        self.seenpix_for_plot = covnorm > 0
        self.mask = np.ones(12 * self.params["SKY"]["nside"] ** 2)
        self.mask[self.seenpix] = self.params["PLANCK"]["weight_planck"]

        ### Angular resolutions
        self._get_convolution()

        self.external_timeline = ExternalData2Timeline(
            self.skyconfig,
            self.joint.qubic.allnus,
            self.params["QUBIC"]["nrec"],
            nside=self.params["SKY"]["nside"],
            corrected_bandpass=self.params["QUBIC"]["bandpass_correction"],
        )

        ### Define reconstructed and TOD operator
        self._get_H()

        ### Inverse noise covariance matrix
        self.invN = self.joint.get_invntt_operator(mask=self.mask)

        ### Noises
        seed_noise_planck = self._get_random_value()
        # print('seed_noise_planck', seed_noise_planck)

        self.noise143 = (
            self.planck_acquisition143.get_noise(seed_noise_planck)
            * self.params["PLANCK"]["level_noise_planck"]
        )
        self.noise217 = (
            self.planck_acquisition217.get_noise(seed_noise_planck + 1)
            * self.params["PLANCK"]["level_noise_planck"]
        )

        if self.params["QUBIC"]["instrument"] == "DB":
            qubic_noise = QubicDualBandNoise(
                self.dict_out,
                self.params["QUBIC"]["npointings"],
                self.params["QUBIC"]["NOISE"]["detector_nep"],
            )
        elif self.params["QUBIC"]["instrument"] == "UWB":
            qubic_noise = QubicWideBandNoise(
                self.dict_out,
                self.params["QUBIC"]["npointings"],
                self.params["QUBIC"]["NOISE"]["detector_nep"],
            )

        self.noiseq = qubic_noise.total_noise(
            self.params["QUBIC"]["NOISE"]["ndet"],
            self.params["QUBIC"]["NOISE"]["npho150"],
            self.params["QUBIC"]["NOISE"]["npho220"],
            seed_noise=seed_noise_planck,
        ).ravel()

        ### Initialize plot instance
        self.plots = PlotsFMM(self.seenpix)

    def _get_components_fgb(self):

        comps = []

        if self.params["CMB"]["cmb"]:
            comps += [CMB()]

        if self.params["Foregrounds"]["Dust"]:
            comps += [Dust(nu0=150, beta_d=1.54, temp=20)]

        if self.params["Foregrounds"]["Synchrotron"]:
            comps += [Synchrotron(nu0=150, beta_pl=-3)]

        return comps

    def _get_random_value(self):

        np.random.seed(None)
        if self.rank == 0:
            seed = np.random.randint(10000000)
        else:
            seed = None

        seed = self.comm.bcast(seed, root=0)
        return seed

    def _get_H(self):
        """

        Method to compute QUBIC operators.

        """

        ### Pointing matrix for TOD generation
        self.H_in = self.joint_tod.get_operator(
            fwhm=self.fwhm_in
        )  # , external_data=self.params['PLANCK']['external_data'])

        ### QUBIC Pointing matrix for TOD generation
        self.H_in_qubic = self.joint_tod.qubic.get_operator(fwhm=self.fwhm_in)

        ### Pointing matrix for reconstruction
        self.H_out_all_pix = self.joint.get_operator(fwhm=self.fwhm_out)
        self.H_out = self.joint.get_operator(
            fwhm=self.fwhm_out, seenpix=self.seenpix
        )  # , external_data=self.params['PLANCK']['external_data'])

    def _get_averaged_nus(self):
        """

        Method to average QUBIC frequencies.

        """

        nus_eff = []
        for i in range(self.params["QUBIC"]["nrec"]):
            nus_eff += [
                np.mean(self.joint.qubic.allnus[i * self.fsub : (i + 1) * self.fsub])
            ]

        return np.array(nus_eff)

    def _get_sky_config(self):
        """

        Method that read `params.yml` file and create dictionary containing sky emission such as :

                    d = {'cmb':seed, 'dust':'d0', 'synchrotron':'s0'}

        Note that the key denote the emission and the value denote the sky model using PySM convention. For CMB, seed denote the realization.

        """

        sky = {}

        if self.params["CMB"]["cmb"]:
            if self.params["CMB"]["seed"] == 0:
                if self.rank == 0:
                    seed = np.random.randint(10000000)
                else:
                    seed = None
                seed = self.comm.bcast(seed, root=0)
            else:
                seed = self.params["CMB"]["seed"]
            print(f"Seed of the CMB is {seed} for rank {self.rank}")
            sky["cmb"] = seed

        for j in self.params["Foregrounds"]:
            # print(j, self.params['Foregrounds'][j])
            if j == "Dust":
                if self.params["Foregrounds"][j]:
                    sky["dust"] = "d0"
            elif j == "Synchrotron":
                if self.params["Foregrounds"][j]:
                    sky["synchrotron"] = "s0"

        return sky

    def get_ultrawideband_config(self):
        """

        Method that pre-compute UWB configuration.

        """

        nu_up = 247.5
        nu_down = 131.25
        nu_ave = np.mean(np.array([nu_up, nu_down]))
        delta = nu_up - nu_ave

        return nu_ave, 2 * delta / nu_ave

    def get_dict(self, key="in"):
        """

        Method to modify the qubic dictionary.

        """

        nu_ave, delta_nu_over_nu = self.get_ultrawideband_config()

        args = {
            "npointings": self.params["QUBIC"]["npointings"],
            "nf_recon": self.params["QUBIC"]["nrec"],
            "nf_sub": self.params["QUBIC"][f"nsub_{key}"],
            "nside": self.params["SKY"]["nside"],
            "MultiBand": True,
            "period": 1,
            "RA_center": self.params["SKY"]["RA_center"],
            "DEC_center": self.params["SKY"]["DEC_center"],
            "filter_nu": 150 * 1e9,
            "noiseless": False,
            "comm": self.comm,
            "dtheta": self.params["QUBIC"]["dtheta"],
            "nprocs_sampling": 1,
            "nprocs_instrument": self.size,
            "photon_noise": True,
            "nhwp_angles": 3,
            #'effective_duration':3,
            "effective_duration150": 3,
            "effective_duration220": 3,
            "filter_relative_bandwidth": 0.25,
            "type_instrument": "wide",
            "TemperatureAtmosphere150": None,
            "TemperatureAtmosphere220": None,
            "EmissivityAtmosphere150": None,
            "EmissivityAtmosphere220": None,
            "detector_nep": float(self.params["QUBIC"]["NOISE"]["detector_nep"]),
            "synthbeam_kmax": self.params["QUBIC"]["SYNTHBEAM"]["synthbeam_kmax"],
        }

        args_mono = args.copy()
        args_mono["nf_recon"] = 1
        args_mono["nf_sub"] = 1

        ### Get the default dictionary
        dictfilename = "dicts/pipeline_demo.dict"
        d = qubic.qubicdict.qubicDict()
        d.read_from_file(dictfilename)
        dmono = d.copy()
        for i in args.keys():

            d[str(i)] = args[i]
            dmono[str(i)] = args_mono[i]

        return d, dmono

    def _get_convolution(self):
        """

        Method to define expected QUBIC angular resolutions (radians) as function of frequencies.

        """

        ### Define FWHMs

        self.fwhm_in = None
        self.fwhm_out = None

        ### FWHMs during map-making
        if self.params["QUBIC"]["convolution_in"]:
            self.fwhm_in = self.joint.qubic.allfwhm.copy()
        if self.params["QUBIC"]["convolution_out"]:
            self.fwhm_out = np.array([])
            for irec in range(self.params["QUBIC"]["nrec"]):
                self.fwhm_out = np.append(
                    self.fwhm_out,
                    np.sqrt(
                        self.joint.qubic.allfwhm[
                            irec * self.fsub : (irec + 1) * self.fsub
                        ]
                        ** 2
                        - np.min(
                            self.joint.qubic.allfwhm[
                                irec * self.fsub : (irec + 1) * self.fsub
                            ]
                        )
                        ** 2
                    ),
                )

        ### Define reconstructed FWHM depending on the user's choice
        if (
            self.params["QUBIC"]["convolution_in"]
            and self.params["QUBIC"]["convolution_out"]
        ):
            self.fwhm_rec = np.array([])
            for irec in range(self.params["QUBIC"]["nrec"]):
                self.fwhm_rec = np.append(
                    self.fwhm_rec,
                    np.min(
                        self.joint.qubic.allfwhm[
                            irec * self.fsub : (irec + 1) * self.fsub
                        ]
                    ),
                )

        elif (
            self.params["QUBIC"]["convolution_in"]
            and self.params["QUBIC"]["convolution_out"] is False
        ):
            self.fwhm_rec = np.array([])
            for irec in range(self.params["QUBIC"]["nrec"]):
                self.fwhm_rec = np.append(
                    self.fwhm_rec,
                    np.mean(
                        self.joint.qubic.allfwhm[
                            irec * self.fsub : (irec + 1) * self.fsub
                        ]
                    ),
                )

        else:
            self.fwhm_rec = np.array([])
            for irec in range(self.params["QUBIC"]["nrec"]):
                self.fwhm_rec = np.append(self.fwhm_rec, 0)

        if self.rank == 0:
            print(f"FWHM for TOD generation : {self.fwhm_in}")
            print(f"FWHM for reconstruction : {self.fwhm_out}")
            print(f"Final FWHM : {self.fwhm_rec}")

    def get_input_map(self):
        m_nu_in = np.zeros(
            (self.params["QUBIC"]["nrec"], 12 * self.params["SKY"]["nside"] ** 2, 3)
        )

        for i in range(self.params["QUBIC"]["nrec"]):
            m_nu_in[i] = np.mean(
                self.external_timeline.m_nu[i * self.fsub : (i + 1) * self.fsub], axis=0
            )

        return m_nu_in

    def _get_tod(self, noise=False):
        """

        Method that compute observed TODs with TOD = H . s + n with H the QUBIC operator, s the sky signal and n the instrumental noise.

        """

        if noise:
            factor = 0
        else:
            factor = 1
        if self.params["QUBIC"]["instrument"] == "UWB":
            if self.params["QUBIC"]["nrec"] != 1:
                TOD_PLANCK = np.zeros(
                    (
                        self.params["QUBIC"]["nrec"],
                        12 * self.params["SKY"]["nside"] ** 2,
                        3,
                    )
                )
                for irec in range(int(self.params["QUBIC"]["nrec"] / 2)):
                    if self.params["QUBIC"]["convolution_in"]:
                        C = HealpixConvolutionGaussianOperator(
                            fwhm=np.min(
                                self.fwhm_in[irec * self.fsub : (irec + 1) * self.fsub]
                            )
                        )

                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)

                    TOD_PLANCK[irec] = C(
                        factor * self.external_timeline.maps[irec] + self.noise143
                    )

                for irec in range(
                    int(self.params["QUBIC"]["nrec"] / 2), self.params["QUBIC"]["nrec"]
                ):
                    if self.params["QUBIC"]["convolution_in"]:
                        C = HealpixConvolutionGaussianOperator(
                            fwhm=np.min(
                                self.fwhm_in[irec * self.fsub : (irec + 1) * self.fsub]
                            )
                        )

                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)

                    TOD_PLANCK[irec] = C(
                        factor * self.external_timeline.maps[irec] + self.noise217
                    )
            else:
                TOD_PLANCK = np.zeros(
                    (
                        2 * self.params["QUBIC"]["nrec"],
                        12 * self.params["SKY"]["nside"] ** 2,
                        3,
                    )
                )

                if self.params["QUBIC"]["convolution_in"]:
                    C = HealpixConvolutionGaussianOperator(fwhm=self.fwhm_in[-1])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)

                TOD_PLANCK[0] = C(
                    factor * self.external_timeline.maps[0] + self.noise143
                )
                TOD_PLANCK[1] = C(
                    factor * self.external_timeline.maps[0] + self.noise217
                )

            TOD_PLANCK = TOD_PLANCK.ravel()
            TOD_QUBIC = (
                self.H_in_qubic(factor * self.external_timeline.m_nu).ravel()
                + self.noiseq
            )
            if self.params["PLANCK"]["external_data"]:
                TOD = np.r_[TOD_QUBIC, TOD_PLANCK]
            else:
                TOD = TOD_QUBIC

        else:

            sh_q = self.joint.qubic.ndets * self.joint.qubic.nsamples
            TOD_QUBIC = (
                self.H_in_qubic(factor * self.external_timeline.m_nu).ravel()
                + self.noiseq
            )
            if self.params["PLANCK"]["external_data"] == False:
                TOD = TOD_QUBIC
            else:
                TOD_QUBIC150 = TOD_QUBIC[:sh_q].copy()
                TOD_QUBIC220 = TOD_QUBIC[sh_q:].copy()

                TOD = TOD_QUBIC150.copy()
                TOD_PLANCK = np.zeros(
                    (
                        self.params["QUBIC"]["nrec"],
                        12 * self.params["SKY"]["nside"] ** 2,
                        3,
                    )
                )
                for irec in range(int(self.params["QUBIC"]["nrec"] / 2)):
                    if self.params["QUBIC"]["convolution_in"]:
                        C = HealpixConvolutionGaussianOperator(
                            fwhm=np.min(
                                self.fwhm_in[irec * self.fsub : (irec + 1) * self.fsub]
                            )
                        )

                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)

                    TOD = np.r_[
                        TOD,
                        C(
                            factor * self.external_timeline.maps[irec] + self.noise143
                        ).ravel(),
                    ]

                TOD = np.r_[TOD, TOD_QUBIC220.copy()]
                for irec in range(
                    int(self.params["QUBIC"]["nrec"] / 2), self.params["QUBIC"]["nrec"]
                ):
                    if self.params["QUBIC"]["convolution_in"]:
                        C = HealpixConvolutionGaussianOperator(
                            fwhm=np.min(
                                self.fwhm_in[irec * self.fsub : (irec + 1) * self.fsub]
                            )
                        )

                    else:
                        C = HealpixConvolutionGaussianOperator(fwhm=0)

                    TOD = np.r_[
                        TOD,
                        C(
                            factor * self.external_timeline.maps[irec] + self.noise217
                        ).ravel(),
                    ]

        self.m_nu_in = self.get_input_map()

        return TOD

    def _barrier(self):
        """

        Method to introduce comm.Barrier() function if MPI communicator is detected.

        """
        if self.comm is None:
            pass
        else:
            self.comm.Barrier()

    def print_message(self, message):
        """

        Method to print message only on rank 0 if MPI communicator is detected. It display simple message if not.

        """

        if self.comm is None:
            print(message)
        else:
            if self.rank == 0:
                print(message)

    def _get_preconditionner(self):

        if self.params["PCG"]["preconditioner"]:

            comps = self._get_components_fgb()
            A = MixingMatrix(*comps).eval(self.joint.qubic.allnus).sum(axis=1)

            approx_hth = np.zeros(
                (
                    self.params["QUBIC"]["nsub_out"],
                    12 * self.params["SKY"]["nside"] ** 2,
                    3,
                )
            )
            conditionner = np.zeros(
                (self.params["QUBIC"]["nrec"], 12 * self.params["SKY"]["nside"] ** 2, 3)
            )
            vec = np.ones(self.joint.qubic.H[0].shapein)

            for i in range(self.params["QUBIC"]["nsub_out"]):
                for j in range(self.params["QUBIC"]["nsub_out"]):
                    approx_hth[i] = (
                        self.joint.qubic.H[i].T
                        * self.joint.qubic.invn220
                        * self.joint.qubic.H[j](vec)
                    )

            for irec in range(self.params["QUBIC"]["nrec"]):
                imin = irec * self.fsub
                imax = (irec + 1) * self.fsub
                for istk in range(3):
                    conditionner[irec, self.seenpix, istk] = 1 / (
                        np.sum(approx_hth[imin:imax, self.seenpix, 0], axis=0)
                    )

            conditionner[conditionner == np.inf] = 1

            M = DiagonalOperator(conditionner[:, self.seenpix, :])
        else:
            M = None
        return M

    def _pcg(self, d, x0, seenpix):
        """

        Solve the map-making equation iteratively :     (H^T . N^{-1} . H) . x = H^T . N^{-1} . d

        The PCG used for the minimization is intrinsequely parallelized (e.g see PyOperators).

        """

        ### Update components when pixels outside the patch are fixed (assumed to be 0)
        A = self.H_out.T * self.invN * self.H_out

        x_planck = self.m_nu_in * (1 - seenpix[None, :, None])
        b = self.H_out.T * self.invN * (d - self.H_out_all_pix(x_planck))
        ### Preconditionning
        M = self._get_preconditionner()

        if self.params["PCG"]["gif"]:
            gif_folder = self.plot_folder + f"{self.job_id}/iter/"
        else:
            gif_folder = None

        ### PCG
        solution_qubic_planck = pcg(
            A=A,
            b=b,
            comm=self.comm,
            x0=x0,
            M=M,
            tol=self.params["PCG"]["tol_pcg"],
            disp=True,
            maxiter=self.params["PCG"]["n_iter_pcg"],
            gif_folder=gif_folder,
            job_id=self.job_id,
            seenpix=self.seenpix,
            seenpix_plot=self.seenpix,
            center=self.center,
            reso=self.params["PCG"]["resolution_plot"],
            fwhm_plot=self.params["PCG"]["fwhm_plot"],
            input=self.m_nu_in,
        )

        if self.params["PCG"]["gif"]:
            do_gif(gif_folder, "iter_", output="animation.gif")

        self._barrier()

        if self.params["QUBIC"]["nrec"] == 1:
            solution_qubic_planck["x"]["x"] = np.array(
                [solution_qubic_planck["x"]["x"]]
            )

        solution = np.ones(self.m_nu_in.shape) * hp.UNSEEN
        solution[:, seenpix, :] = solution_qubic_planck["x"]["x"].copy()

        return solution

    def save_data(self, name, d):
        """

        Method to save data using pickle convention.

        """

        with open(name, "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self):
        """

        Method to run the whole pipeline from TOD generation from sky reconstruction by reading `params.yml` file.

        """

        self.print_message("\n=========== Map-Making ===========\n")

        ### Get simulated data
        self.TOD = self._get_tod(noise=False)
        self.n = self._get_tod(noise=True)

        ### Wait for all processes
        self._barrier()

        if self.params["QUBIC"]["convolution_in"]:
            for i in range(self.params["QUBIC"]["nrec"]):
                C = HealpixConvolutionGaussianOperator(fwhm=self.fwhm_rec[i])
                self.m_nu_in[i] = C(self.m_nu_in[i])

        x0 = self.m_nu_in.copy()
        x0[:, self.seenpix, :] = 0

        ### Solve map-making equation
        self.s_hat = self._pcg(
            self.TOD, x0=x0[:, self.seenpix, :], seenpix=self.seenpix
        )

        ### Wait for all processes
        self._barrier()

        ### n = m_signalnoisy - m_signal
        self.s_hat_noise = self.s_hat - self.m_nu_in

        ### Ensure that non seen pixels is 0 for spectrum computation
        self.s_hat[:, ~self.seenpix, :] = 0
        self.s_hat_noise[:, ~self.seenpix, :] = 0

        ### Plots and saving
        if self.rank == 0:

            self.external_maps = self.externaldata.maps.copy()
            self.external_maps[:, ~self.seenpix, :] = 0

            self.external_maps_noise = self.externaldata_noise.maps.copy()
            self.external_maps_noise[:, ~self.seenpix, :] = 0

            if len(self.externaldata.external_nus) != 0:
                fwhm_ext = self.externaldata.fwhm_ext.copy()
                self.s_hat = np.concatenate((self.s_hat, self.external_maps), axis=0)
                self.s_hat_noise = np.concatenate(
                    (self.s_hat_noise, self.external_maps_noise), axis=0
                )
                self.nus_rec = np.array(
                    list(self.nus_Q) + list(self.externaldata.external_nus)
                )
                self.fwhm_rec = np.array(list(self.fwhm_rec) + list(fwhm_ext))

            self.plots.plot_frequency_maps(
                self.m_nu_in[: len(self.nus_Q)],
                self.s_hat[: len(self.nus_Q)],
                self.center,
                reso=15,
                nsig=3,
                filename=self.plot_folder + f"/all_maps.png",
                figsize=(10, 5),
            )
            # self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=0, nsig=3, name='signal')
            # self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=1, nsig=3, name='signal')
            # self.plots.plot_FMM(self.m_nu_in, self.s_hat, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=2, nsig=3, name='signal')

            # self.plots.plot_FMM(self.m_nu_in*0, self.s_hat_noise, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=0, nsig=3, name='noise')
            # self.plots.plot_FMM(self.m_nu_in*0, self.s_hat_noise, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=1, nsig=3, name='noise')
            # self.plots.plot_FMM(self.m_nu_in*0, self.s_hat_noise, self.center, self.seenpix, self.nus_Q, job_id=self.job_id, istk=2, nsig=3, name='noise')

            mapmaking_time = time.time() - self.mapmaking_time_0
            if self.comm is None:
                print(f"Map-making done in {mapmaking_time:.3f} s")
            else:
                if self.rank == 0:
                    print(f"Map-making done in {mapmaking_time:.3f} s")

            dict_solution = {
                "maps": self.s_hat,
                "maps_noise": self.s_hat_noise,
                "nus": self.nus_rec,
                "coverage": self.coverage,
                "center": self.center,
                "maps_in": self.m_nu_in,
                "parameters": self.params,
                "fwhm_in": self.fwhm_in,
                "fwhm_out": self.fwhm_out,
                "fwhm_rec": self.fwhm_rec,
                "duration": mapmaking_time,
            }

            save_data(self.file, dict_solution)
        self._barrier()


class PipelineEnd2End:
    """

    Wrapper for End-2-End pipeline. It added class one after the others by running method.run().

    """

    def __init__(self, comm):

        with open("src/FMM/params.yml", "r") as stream:
            self.params = yaml.safe_load(stream)

        self.comm = comm
        self.job_id = os.environ.get("SLURM_JOB_ID")

        self.folder = "src/FMM/" + self.params["path_out"] + "maps/"
        self.file = self.folder + self.params["datafilename"] + f"_{self.job_id}.pkl"
        self.file_spectrum = (
            "src/FMM/"
            + self.params["path_out"]
            + "spectrum/"
            + "spectrum_"
            + self.params["datafilename"]
            + f"_{self.job_id}.pkl"
        )
        self.mapmaking = None

    def main(self, specific_file=None):

        ### Execute Frequency Map-Making
        if self.params["Pipeline"]["mapmaking"]:

            ### Initialization
            self.mapmaking = PipelineFrequencyMapMaking(
                self.comm, self.file, self.params
            )

            ### Run
            self.mapmaking.run()

        ### Execute spectrum
        if self.params["Pipeline"]["spectrum"]:
            if self.comm.Get_rank() == 0:
                create_folder_if_not_exists(
                    self.comm, "src/FMM/" + self.params["path_out"] + "spectrum/"
                )

                if self.mapmaking is not None:
                    self.spectrum = Spectra(self.file)
                else:
                    self.spectrum = Spectra(specific_file)

                ### Signal
                DlBB_maps = self.spectrum.run(maps=self.spectrum.maps)

                ### noise
                DlBB_noise = self.spectrum.run(
                    maps=self.spectrum.dictionary["maps_noise"]
                )

                dict_solution = {
                    "nus": self.spectrum.dictionary["nus"],
                    "ell": self.spectrum.ell,
                    "Dls": DlBB_maps,
                    "Nl": DlBB_noise,
                    "parameters": self.params,
                }

                save_data(self.file_spectrum, dict_solution)
