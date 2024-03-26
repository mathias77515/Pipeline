import numpy as np
import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
from qubic import NamasterLib as nam
import qubic
import matplotlib.pyplot as plt

def get_cl_cmb(ell, r, Alens):
        
    power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        
    if Alens != 1.:
        power_spectrum[2] *= Alens
    
    if r:
        power_spectrum += r * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        
    return power_spectrum


NSIDE = 256
FSKY = 0.01
cls = get_cl_cmb(np.arange(1, 4001, 1), 0, 1)
np.random.seed(1)
cmb = hp.synfast(cls, NSIDE, verbose=False, new=True).T

def get_coverage(fsky, nside, center_radec):
    center = qubic.equ2gal(center_radec[0], center_radec[1])
    uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
    uvpix = np.array(hp.pix2vec(nside, np.arange(12*nside**2)))
    ang = np.arccos(np.dot(uvcenter, uvpix))
    indices = np.argsort(ang)
    okpix = ang < -1
    okpix[indices[0:int(fsky * 12*nside**2)]] = True
    mask = np.zeros(12*nside**2)
    mask[okpix] = 1
    return mask

sky = pysm3.Sky(nside=NSIDE, preset_strings=["d0"])
nu0 = 150
map = np.array(sky.get_emission(nu0 * u.GHz, None).T * utils.bandpass_unit_conversion(nu0 * u.GHz, None, u.uK_CMB)).T + cmb.T


def _get_spectra(map, apo):
    print(apo)
    cov = get_coverage(FSKY, NSIDE, center_radec=[0, -57])
    cov = np.array([cov]*3)
    map *= cov
    namaster = nam.Namaster(cov[0], lmin=40, lmax=2*NSIDE-1, delta_ell=30, aposize=apo)
    ell, Dl, _ = namaster.get_spectra(map=map, purify_e=False, purify_b=True, beam_correction=0, pixwin_correction=False, verbose=False)
    index = np.where((ell > 40) & (ell < 500))[0]

    ell = ell[index]
    #print(ell)
    Dl = Dl[index, 2]
    return ell, Dl

fsky = [10]
c = ['darkblue', 'red', 'blue', 'green', 'magenta']

plt.figure(figsize=(8, 8))
m = 0
for ii, i in enumerate(fsky):
    ell, Dl = _get_spectra(map, i)
    print(Dl)
    plt.plot(ell, Dl, '-o', color=c[ii])
    #if np.max(Dl) > m:
    #    m = np.max(Dl)
#plt.ylim(0, m)
plt.savefig('Dl.png')
plt.close()