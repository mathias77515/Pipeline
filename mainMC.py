import numpy as np
import yaml

from pipeline import *
from pyoperators import *

if __name__ == "__main__":

    ### Common MPI arguments
    comm = MPI.COMM_WORLD

    ### Initialization
    pipeline = PipelineMonteCarloCrossSpectrum(comm)

    ### Execution
    pipeline.main()

