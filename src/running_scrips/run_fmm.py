import sys
from pyoperators import *

from ..FMM.pipeline import PipelineEnd2End
#from .FMM.pipeline import PipelineEnd2End


stop
try:
    file = str(sys.argv[1])
except IndexError:
    file = None

if __name__ == "__main__":

    ### Common MPI arguments
    comm = MPI.COMM_WORLD

    ### Initialization
    pipeline = PipelineEnd2End(comm)

    ### Execution
    pipeline.main(specific_file=file)
