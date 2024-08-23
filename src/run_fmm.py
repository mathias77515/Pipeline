from FMM.pipeline import PipelineEnd2End
from pyoperators import *
import sys

try:
    file = str(sys.argv[1])
except IndexError:
    file = 1
    
if __name__ == "__main__":

    ### Common MPI arguments
    comm = MPI.COMM_WORLD

    ### Initialization
    pipeline = PipelineEnd2End(comm)

    ### Execution
    pipeline.main(specific_file=file)

