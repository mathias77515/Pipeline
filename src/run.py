#import sys
#from pyoperators import *
#from FMM.pipeline import PipelineEnd2End
#from CMM.pipeline import Pipeline
import running_scripts
stop
simu = 'CMM'

if __name__ == "__main__":

    ### Common MPI arguments
    comm = MPI.COMM_WORLD

    
    if simu == 'FMM':
        
        try:
            file = str(sys.argv[1])
        except IndexError:
            file = None

        ### Initialization
        pipeline = PipelineEnd2End(comm)
        
        ### Execution
        pipeline.main(specific_file=file)

    elif simu == 'CMM':

        seed_noise = int(sys.argv[1])
        
        ### Initialization
        pipeline = Pipeline(comm, 1, seed_noise)
        
        ### Execution
        pipeline.main()

    