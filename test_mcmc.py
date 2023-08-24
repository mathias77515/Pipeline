import numpy as np
import yaml

from pipeline import *
from pyoperators import *
from model.externaldata import *
import pysm3
import pysm3.units as u
from pysm3 import utils

with open('data.pkl', 'wb') as handle:
    pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)

### Initialization
externaldata = PipelineExternalData({'cmb':4}, 'data.pkl')

### Run -> compute frequency maps
externaldata.run(fwhm=False, noise=False)

### Update data.pkl -> Add frequency maps
externaldata._update_data(externaldata.maps, externaldata.external_nus)

### Initialization
spectrum = Spectrum('data.pkl')

### Run -> compute Dl-cross
spectrum.run()

### Update data.pkl -> Add spectrum
spectrum._update_data(spectrum.ell, spectrum.Dl)
#print(spectrum._read_pkl())

### Initialization
cross = PipelineCrossSpectrum('data.pkl')

### Run
cross.run()






