import numpy as np
import yaml

from pipeline import *
from pyoperators import *

### Common MPI arguments
comm = MPI.COMM_WORLD

### Initialization
pipeline = PipelineFrequencyMapMaking(comm)

### Run pipeline
pipeline.run()












#