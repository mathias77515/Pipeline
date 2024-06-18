from pipeline import *
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
<<<<<<< HEAD
    pipeline.main(specific_file=file)
=======
    pipeline.main(specific_file=file)

>>>>>>> a30ac34a23ac31b1a9458d6eee3ede980e348ea4
